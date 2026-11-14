import argparse
from pathlib import Path

import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_DIR = Path(__file__).resolve().parent / "models" / "wav2vec2-emotion"
SAMPLE_RATE = 16_000
NUM_SAMPLES = 4 * SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from a speech WAV.")
    parser.add_argument("audio", type=Path, help="Path to a WAV file")
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")
    if not MODEL_DIR.is_dir():
        parser.error(f"Copy the saved Colab model folder to: {MODEL_DIR}")

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model.eval()

    waveform, _ = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True)
    waveform = librosa.util.fix_length(waveform, size=NUM_SAMPLES)
    inputs = extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.inference_mode():
        logits = model(**inputs).logits
        label_id = logits.argmax(dim=-1).item()

    print("Predicted emotion:", model.config.id2label[label_id])


if __name__ == "__main__":
    main()