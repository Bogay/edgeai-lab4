import io
import sys
import time
from jiwer import wer
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class Row:
    audio_duration: float
    inference_duration: float
    reference: str
    output: str

def parse_line(l: str) -> Row:
    split = l.strip().split("\t")
    # print(split)
    return Row(
        audio_duration=float(split[0]),
        inference_duration=float(split[1]),
        reference=split[2],
        output=split[3],
    )


if __name__ == "__main__":
    inputs = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore')
    rows = [parse_line(l) for l in inputs if l.strip()]

    predictions = [r.output for r in rows]
    references = [r.reference for r in rows]
    total_audio_duration = sum(r.audio_duration for r in rows)
    total_inference_duration = sum(r.inference_duration for r in rows)
    
    error = wer(references, predictions)
    rtime_factor = total_inference_duration / total_audio_duration

    print(f"WER: {error:.4f}")
    print(f"Total time: {total_inference_duration:.2f} sec")
    print(f"Real-time factor (RTF): {rtime_factor:.4f}")

