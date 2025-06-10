use hound::WavReader;
use indicatif::{ProgressBar, ProgressStyle};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// Custom error type for better error handling
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// --- Helper Functions ---

/// Calculates Character Error Rate (CER) between two strings.
/// This is a basic implementation of Levenshtein distance for CER.
/// It might not be as robust as `jiwer.cer` but provides a reasonable approximation.
fn calculate_cer(reference: &str, hypothesis: &str) -> f64 {
    let ref_chars: Vec<char> = reference.chars().collect();
    let hyp_chars: Vec<char> = hypothesis.chars().collect();

    let n = ref_chars.len();
    let m = hyp_chars.len();

    // Create a DP table to store minimum edit distances
    let mut dp = vec![vec![0; m + 1]; n + 1];

    // Initialize base cases
    for i in 0..=n {
        dp[i][0] = i; // Cost of deleting all reference characters
    }
    for j in 0..=m {
        dp[0][j] = j; // Cost of inserting all hypothesis characters
    }

    // Fill the DP table
    for i in 1..=n {
        for j in 1..=m {
            let cost = if ref_chars[i - 1] == hyp_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1) // Deletion
                .min(dp[i][j - 1] + 1) // Insertion
                .min(dp[i - 1][j - 1] + cost); // Substitution
        }
    }

    let distance = dp[n][m];

    // Handle division by zero if reference is empty
    if n == 0 {
        return if m == 0 { 0.0 } else { 1.0 };
    }

    (distance as f64) / (n as f64)
}

/// Loads a WAV file and converts its samples to f32.
/// Whisper.cpp expects 16kHz mono audio. If the input is different,
/// it might lead to poor results or errors. This function will warn the user.
fn load_wav_to_f32(path: &Path) -> Result<Vec<f32>> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();

    // Convert i16 samples (typical WAV format) to f32, normalizing to -1.0 to 1.0
    // This is the format whisper.cpp's `full` method expects.
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.map(|val| val as f32 / 32768.0).unwrap_or(0.0)) // Handle potential sample read errors
        .collect();

    // Check if the audio meets Whisper's expectations (16kHz, mono)
    if spec.sample_rate != 16000 || spec.channels != 1 {
        eprintln!(
            "Warning: Audio file at {:?} is not 16kHz mono. \
             Whisper expects 16kHz mono audio. Performance/accuracy may be affected.",
            path
        );
        // For a robust application, you'd integrate a resampling library here
        // (e.g., `rubato`). For this benchmark, we proceed as is.
        // let params = SincInterpolationParameters {
        //     sinc_len: 256,
        //     f_cutoff: 0.95,
        //     interpolation: SincInterpolationType::Linear,
        //     oversampling_factor: 256,
        //     window: rubato::WindowFunction::BlackmanHarris,
        // };
        // let channels = 1;
        // let f_ratio = 16000f64 / spec.sample_rate as f64;
        // let mut resampler = SincFixedIn::<f32>::new(f_ratio, 1.1, params, 1024, channels).unwrap();

        // // Prepare
        // let mut input_frames_next = resampler.input_frames_next();
        // // let resampler_delay = resampler.output_delay();
        // let mut outbuffer = vec![vec![0.0f32; resampler.output_frames_max()]; channels];
        // let indata_slices = vec![&samples];
        // while indata_slices[0].len() >= input_frames_next {
        //     resampler.process_into_buffer(&indata_slices, &mut outbuffer, None)?;
        //     input_frames_next = resampler.input_frames_next();
        // }

        // if !indata_slices[0].is_empty() {
        //     resampler.process_into_buffer(&indata_slices, &mut outbuffer, None)?;
        // }
        // return Ok(outbuffer[0].clone());
    }

    Ok(samples)
}

// --- Main Benchmark Logic ---

fn main() -> Result<()> {
    // --- Configuration ---
    // IMPORTANT:
    // 1. Download a Whisper model (e.g., `ggml-base.bin` or `ggml-tiny.bin`)
    //    from the official `ggerganov/whisper.cpp` GitHub releases.
    //    Example: https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin
    // 2. Create a directory named `models` in your project root.
    // 3. Place the downloaded model file inside the `models` directory.
    let model_path = "models/ggml-tiny.bin";

    let audio_dir = Path::new("audio");
    let references_file = Path::new("references.txt");
    if !audio_dir.exists() || !references_file.exists() {
        return Err("'audio/' directory or 'references.txt' not found.")?;
    }

    // Load actual audio files and references if directories exist
    let refs_content = fs::read_to_string(references_file)?;
    let references: Vec<String> = refs_content.lines().map(|s| s.to_string()).collect();

    let mut audio_samples: Vec<(PathBuf, String)> = Vec::new();
    let mut audio_files: Vec<PathBuf> = fs::read_dir(audio_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("wav") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    audio_files.sort(); // Ensure consistent order, important for matching with references
    for (i, audio_path) in audio_files.iter().enumerate() {
        if let Some(reference) = references.get(i) {
            audio_samples.push((audio_path.clone(), reference.clone()));
        } else {
            eprintln!(
                "Warning: No reference found for audio file at index {}: {:?}",
                i, audio_path
            );
        }
    }
    if audio_samples.is_empty() {
        return Err("No valid audio files and references found for benchmarking. Please ensure data is prepared correctly or dummy WAVs exist.".into());
    }

    // Calculate total audio duration
    let mut total_audio_duration_seconds = 0.0f64;
    for (audio_path, _) in &audio_samples {
        let reader = WavReader::open(audio_path)?;
        let spec = reader.spec();
        let duration_frames = reader.len() as f64;
        total_audio_duration_seconds += duration_frames / spec.sample_rate as f64;
    }

    // --- Initialize Whisper Context ---
    let ctx_params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(model_path, ctx_params)
        .expect("Failed to load Whisper model. Make sure ggml-base.bin (or your chosen model) is in the 'models/' directory.");

    // --- Warming up the model ---
    println!("Warming up the model...");
    let pb_warmup = ProgressBar::new(3);
    pb_warmup.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")?
            .progress_chars("##-"),
    );

    // Use the first available audio sample for warmup
    let audio_to_warmup = if let Some((path, _)) = audio_samples.first() {
        load_wav_to_f32(path)?
    } else {
        return Err("Failed to load audio file to warmup")?;
    };

    for _ in 0..3 {
        let mut state = ctx.create_state()?;
        let mut full_params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        full_params.set_language(Some("zh")); // Set language to Chinese (zh-TW is not a specific language code for Whisper)
        full_params.set_print_special(false); // Suppress special tokens in output
        full_params.set_print_progress(false); // Suppress internal progress during inference
        full_params.set_print_realtime(false);
        full_params.set_print_timestamps(false);

        state.full(full_params, &audio_to_warmup)?;
        pb_warmup.inc(1);
    }
    pb_warmup.finish_with_message("Warmup complete.");

    // --- Main ASR Processing ---
    println!(
        "Starting ASR processing for {} samples...",
        audio_samples.len()
    );
    let start_time = Instant::now();
    let n_samples = audio_samples.len();
    let mut all_predictions: Vec<String> = Vec::with_capacity(n_samples);
    let mut all_references: Vec<String> = Vec::with_capacity(n_samples);

    let pb_asr = ProgressBar::new(audio_samples.len() as u64);
    pb_asr.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")?
            .progress_chars("##-"),
    );

    for (i, (audio_path, reference_text)) in audio_samples.into_iter().enumerate() {
        // Load audio for the current sample
        let audio_f32 = match load_wav_to_f32(&audio_path) {
            Ok(audio) => audio,
            Err(e) => {
                eprintln!("Error loading audio file {:?}: {}", audio_path, e);
                pb_asr.inc(1); // Still increment progress bar
                continue; // Skip this sample
            }
        };

        let mut state = ctx.create_state()?;
        let mut full_params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        full_params.set_language(Some("zh"));
        full_params.set_print_special(false);
        full_params.set_print_progress(false);
        full_params.set_print_realtime(false);
        full_params.set_print_timestamps(false);

        // Perform transcription
        state.full(full_params, &audio_f32)?;
        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");
        let hypothesis = (0..num_segments)
            .map(|i| {
                state
                    .full_get_segment_text_lossy(i)
                    .expect("failed to get segment")
            })
            .collect::<Vec<_>>()
            .join("");
        all_predictions.push(hypothesis.clone());
        all_references.push(reference_text.clone());

        pb_asr.inc(1);
        pb_asr.set_message(format!("Processed sample {}...", i + 1));
        // You can uncomment the line below for real-time output similar to Python's print
        println!("{reference_text} - {hypothesis}");
    }
    pb_asr.finish_with_message("ASR processing complete.");

    let duration = start_time.elapsed().as_secs_f64();

    // --- Calculate CER and Speed ---
    let total_cer_sum: f64 = all_references
        .iter()
        .zip(all_predictions.iter())
        .map(|(r, p)| calculate_cer(r, p))
        .sum();
    let average_cer = if !all_references.is_empty() {
        total_cer_sum / all_references.len() as f64
    } else {
        0.0
    };

    let rtime_factor = duration / total_audio_duration_seconds;

    println!("\n--- Benchmark Results ---");
    println!(
        "Total audio duration: {:.2} sec",
        total_audio_duration_seconds
    );
    println!("Average CER: {:.4}", average_cer);
    println!("Total transcription time: {:.2} sec", duration);
    println!("Real-time factor (RTF): {:.4}", rtime_factor);

    // Disclaimer regarding OpenCC
    println!(
        "\nNote on CER: This benchmark does NOT include Simplified-to-Traditional Chinese (OpenCC) conversion."
    );
    println!(
        "Whisper's 'chinese' language setting often outputs Simplified Chinese. If your references are Traditional Chinese,"
    );
    println!(
        "the reported CER might be higher due to character differences. For a precise comparison, an OpenCC equivalent in Rust would be needed."
    );

    Ok(())
}
