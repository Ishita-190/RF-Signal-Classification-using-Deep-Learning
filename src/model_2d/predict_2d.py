from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset import get_label_mapping
from model_2d.model_2d import SpectrogramCNN
from model_2d.preprocess_2d import (
    WINDOW_SIZE,
    _pad_to_window_multiple,
    normalize,
    window_to_spectrogram,
)


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PREDICT_SAMPLES_DIR = Path(__file__).resolve().parents[2] / "predict_samples"
BEST_MODEL_PATH = MODELS_DIR / "best_model_2d.pth"


def _load_class_names(dataset_dir: Path | None = None) -> list[str]:
    label_mapping = (
        get_label_mapping(dataset_dir)
        if dataset_dir is not None
        else get_label_mapping()
    )
    return [name for name, _ in sorted(label_mapping.items(), key=lambda item: item[1])]


def load_trained_model(
    device: torch.device | None = None,
    checkpoint_path: Path | str = BEST_MODEL_PATH,
    num_classes: int | None = None,
) -> SpectrogramCNN:
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_classes is None:
        num_classes = len(_load_class_names())
    model = SpectrogramCNN(num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_signal_file(file_path: Path | str) -> np.ndarray:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    return np.load(file_path, allow_pickle=True)


def _extract_label(sample: np.ndarray) -> str | None:
    if isinstance(sample, np.ndarray) and sample.dtype == object and sample.shape == ():
        sample = sample.item()
    if isinstance(sample, dict):
        label = sample.get("label")
        return str(label) if label is not None else None
    return None


def _signal_to_spectrograms(
    signal: np.ndarray, window_size: int = WINDOW_SIZE, hop_length: int | None = None
) -> np.ndarray:
    hop_length = window_size if hop_length is None else hop_length
    padded = _pad_to_window_multiple(normalize(signal), window_size, hop_length)
    window_count = (
        1
        if padded.size <= window_size
        else 1 + (padded.size - window_size) // hop_length
    )
    spectrograms = []
    for i in range(window_count):
        start = i * hop_length
        window = padded[start : start + window_size]
        spectrograms.append(window_to_spectrogram(window))
    return np.stack(spectrograms)[:, np.newaxis, :, :].astype(np.float32)


def predict_windows(
    model: SpectrogramCNN, spectrograms: np.ndarray, device: torch.device | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    tensor = torch.from_numpy(spectrograms).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.cpu().numpy(), torch.argmax(probabilities, dim=1).cpu().numpy()


def predict_file(
    file_path: Path | str,
    model: SpectrogramCNN | None = None,
    device: torch.device | None = None,
    checkpoint_path: Path | str = BEST_MODEL_PATH,
    class_names: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    hop_length: int | None = None,
) -> dict[str, Any]:
    if class_names is None:
        class_names = _load_class_names()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_trained_model(
            device=device, checkpoint_path=checkpoint_path, num_classes=len(class_names)
        )
    else:
        model = model.to(device)
        model.eval()

    raw_sample = load_signal_file(file_path)
    true_label = _extract_label(raw_sample)
    spectrograms = _signal_to_spectrograms(
        raw_sample, window_size=window_size, hop_length=hop_length
    )
    window_probabilities, window_predictions = predict_windows(
        model, spectrograms, device=device
    )
    aggregated = window_probabilities.mean(axis=0)
    final_index = int(np.argmax(aggregated))
    final_class = class_names[final_index]
    final_confidence = float(aggregated[final_index])

    window_results = [
        {
            "window": i + 1,
            "predicted_class": class_names[int(p)],
            "confidence": float(probs[int(p)]),
        }
        for i, (p, probs) in enumerate(zip(window_predictions, window_probabilities))
    ]

    return {
        "file": str(file_path),
        "true_label": true_label,
        "matches_label": true_label == final_class if true_label is not None else None,
        "predicted_class": final_class,
        "confidence": final_confidence,
        "window_predictions": window_results,
        "window_confidences": [r["confidence"] for r in window_results],
        "class_probabilities": {
            name: float(prob) for name, prob in zip(class_names, aggregated)
        },
    }


def predict_path(
    input_path: Path | str = PREDICT_SAMPLES_DIR,
    checkpoint_path: Path | str = BEST_MODEL_PATH,
    window_size: int = WINDOW_SIZE,
    hop_length: int | None = None,
) -> list[dict[str, Any]]:
    input_path = Path(input_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = _load_class_names()
    model = load_trained_model(
        device=device, checkpoint_path=checkpoint_path, num_classes=len(class_names)
    )

    if input_path.is_dir():
        file_paths = sorted(input_path.glob("*.npy"))
    elif input_path.suffix.lower() == ".npy":
        file_paths = [input_path]
    else:
        raise ValueError(f"Expected a .npy file or directory, got: {input_path}")

    return [
        predict_file(
            fp,
            model=model,
            device=device,
            class_names=class_names,
            window_size=window_size,
            hop_length=hop_length,
        )
        for fp in file_paths
    ]


def _format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def print_prediction(result: dict[str, Any]) -> None:
    print(f"File:\n{result['file']}\n")
    print(f"Predicted Class:\n{result['predicted_class']}\n")
    print(f"Confidence:\n{_format_percentage(result['confidence'])}")


def main() -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(
        description="Predict the signal class for one .npy file or a directory of .npy files."
    )
    parser.add_argument("input_path", nargs="?", default=str(PREDICT_SAMPLES_DIR))
    parser.add_argument("--checkpoint", default=str(BEST_MODEL_PATH))
    parser.add_argument("--hop_length", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()

    results = predict_path(
        args.input_path, checkpoint_path=args.checkpoint, hop_length=args.hop_length
    )
    for index, result in enumerate(results):
        if index > 0:
            print()
        print_prediction(result)
    return results


if __name__ == "__main__":
    main()
