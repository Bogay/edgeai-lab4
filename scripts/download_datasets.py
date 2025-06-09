import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    audio_dir = "audio"
    references_file = "references.txt"

    os.makedirs(audio_dir, exist_ok=True)

    print("Loading Common Voice 16.1 zh-TW test split (first 500 samples)...")
    # Load the dataset
    dataset = load_dataset("mozilla-foundation/common_voice_16_1", "zh-TW", split="test[:500]")
    # Filter out samples with missing audio or sentence
    dataset = dataset.filter(lambda x: x["audio"] is not None and x["sentence"] is not None)
    print(f"Found {len(dataset)} valid samples to process.")

    references = []
    print(f"Saving audio files to '{audio_dir}/' and references to '{references_file}'...")
    for i, sample in enumerate(tqdm(dataset)):
        audio_data = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        sentence = sample["sentence"]

        audio_path = os.path.join(audio_dir, f"audio_{i:04}.wav")
        
        # Ensure audio is 16kHz mono, as expected by Whisper
        # `soundfile.write` can handle resampling if needed, but it's better to ensure consistency.
        if sampling_rate != 16000:
            print(f"Warning: Audio sample {i} has sample rate {sampling_rate}, expected 16000. This might impact performance.")
            # For robustness, you might want to resample explicitly here
            # Example: resampled_audio = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=16000)
            # For simplicity, we'll write directly and trust Whisper.cpp's handling.
        
        # Save audio as WAV (16-bit PCM, mono)
        # `soundfile` by default handles conversion from float array to WAV's int16
        sf.write(audio_path, audio_data, 16000, subtype='PCM_16') 
        references.append(sentence)

    with open(references_file, "w", encoding="utf-8") as f:
        for ref in references:
            f.write(ref + "\n")

    print("\nCommon Voice dataset setup complete.")


