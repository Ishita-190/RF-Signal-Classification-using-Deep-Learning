from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset import get_label_mapping
from model_1d.model import SignalCNN
from model_1d.preprocess import (
    WINDOW_SIZE,
    _pad_to_window_multiple,
    iq_to_channels,
    normalize,
)


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PREDICT_SAMPLES_DIR = Path(__file__).resolve().parents[2] / "predict_samples"
BEST_MODEL_PATH = MODELS_DIR / "best_model_1d.pth"


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
) -> SignalCNN:
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_classes is None:
        num_classes = len(_load_class_names())

    model = SignalCNN(num_classes=num_classes).to(device)
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


def _window_signal(
    signal: np.ndarray, window_size: int = WINDOW_SIZE, stride: int | None = None
) -> np.ndarray:
    stride = window_size if stride is None else stride
    signal = _pad_to_window_multiple(normalize(signal), window_size, stride)
    window_count = (
        1 if signal.size <= window_size else 1 + (signal.size - window_size) // stride
    )
    windows = []
    for window_index in range(window_count):
        start = window_index * stride
        window = signal[start : start + window_size]
        windows.append(iq_to_channels(window))
    return np.asarray(windows, dtype=np.float32)


def prepare_input(
    file_path: Path | str, window_size: int = WINDOW_SIZE, stride: int | None = None
) -> tuple[np.ndarray, Path]:
    path = Path(file_path)
    signal = load_signal_file(path)
    windows = _window_signal(signal, window_size=window_size, stride=stride)
    return windows, path


def predict_windows(
    model: SignalCNN,
    windows: np.ndarray,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device

    window_tensor = torch.from_numpy(windows).to(device)
    with torch.no_grad():
        logits = model(window_tensor)
        probabilities = torch.softmax(logits, dim=1)

    return probabilities.cpu().numpy(), torch.argmax(probabilities, dim=1).cpu().numpy()


def aggregate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected a 2D probability array, got shape {probabilities.shape}"
        )
    return probabilities.mean(axis=0)


def predict_file(
    file_path: Path | str,
    model: SignalCNN | None = None,
    device: torch.device | None = None,
    checkpoint_path: Path | str = BEST_MODEL_PATH,
    class_names: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
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
    windows = _window_signal(raw_sample, window_size=window_size, stride=stride)
    path = Path(file_path)
    window_probabilities, window_predictions = predict_windows(
        model, windows, device=device
    )
    aggregated_probabilities = aggregate_probabilities(window_probabilities)
    final_index = int(np.argmax(aggregated_probabilities))
    final_class = class_names[final_index]
    final_confidence = float(aggregated_probabilities[final_index])

    window_results = []
    for window_number, (prediction_index, probabilities) in enumerate(
        zip(window_predictions, window_probabilities), start=1
    ):
        predicted_class = class_names[int(prediction_index)]
        confidence = float(probabilities[int(prediction_index)])
        window_results.append(
            {
                "window": window_number,
                "predicted_class": predicted_class,
                "confidence": confidence,
            },
        )

    return {
        "file": str(path),
        "true_label": true_label,
        "matches_label": true_label == final_class if true_label is not None else None,
        "predicted_class": final_class,
        "confidence": final_confidence,
        "window_predictions": window_results,
        "window_confidences": [result["confidence"] for result in window_results],
        "class_probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(class_names, aggregated_probabilities)
        },
    }


def predict_path(
    input_path: Path | str = PREDICT_SAMPLES_DIR,
    checkpoint_path: Path | str = BEST_MODEL_PATH,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
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

    results = []
    for file_path in file_paths:
        results.append(
            predict_file(
                file_path,
                model=model,
                device=device,
                checkpoint_path=checkpoint_path,
                class_names=class_names,
                window_size=window_size,
                stride=stride,
            )
        )
    return results


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
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(PREDICT_SAMPLES_DIR),
        help="Path to a .npy file or a directory containing .npy files.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(BEST_MODEL_PATH),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=WINDOW_SIZE,
        help="Window stride used during inference. Defaults to the training window size.",
    )
    args = parser.parse_args()

    results = predict_path(
        args.input_path, checkpoint_path=args.checkpoint, stride=args.stride
    )
    for index, result in enumerate(results):
        if index > 0:
            print()
        print_prediction(result)
    return results


if __name__ == "__main__":
    main()
