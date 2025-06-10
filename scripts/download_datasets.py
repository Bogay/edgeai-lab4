import os
import shutil
import soundfile as sf
import torchaudio
from subprocess import check_call
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
        sentence = sample["sentence"]
        audio_path = os.path.join(audio_dir, f"audio_{i:04}.wav")
        raw_audio = sf.read(sample["audio"]["path"])[0]
        if sample["audio"]["sampling_rate"] != 16000:
            waveform, _ = torchaudio.load(sample["audio"]["path"])
            waveform = torchaudio.transforms.Resample(orig_freq=sample["audio"]["sampling_rate"], new_freq=16000)(waveform)
            raw_audio = waveform.numpy()[0]

        sf.write(audio_path, raw_audio, 16000, format='WAV')
        references.append(sentence)

    with open(references_file, "w", encoding="utf-8") as f:
        for ref in references:
            f.write(ref + "\n")

    print("\nCommon Voice dataset setup complete.")
