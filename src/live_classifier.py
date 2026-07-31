from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from rtlsdr import RtlSdr

from live_capture import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SAMPLE_RATE,
    configure_sdr,
    read_iq_samples,
)

from model_1d.predict import (
    BEST_MODEL_PATH as BEST_MODEL_PATH_1D,
    WINDOW_SIZE as WINDOW_SIZE_1D,
    _load_class_names,
    _window_signal,
    aggregate_probabilities,
    load_trained_model as load_trained_model_1d,
    predict_windows as predict_windows_1d,
)
from model_2d.predict_2d import (
    BEST_MODEL_PATH as BEST_MODEL_PATH_2D,
    WINDOW_SIZE as WINDOW_SIZE_2D,
    _signal_to_spectrograms,
    load_trained_model as load_trained_model_2d,
    predict_windows as predict_windows_2d,
)

ModelType = Literal["1d", "2d"]


def _format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def preprocess_iq(
    iq: np.ndarray,
    model_type: ModelType = "1d",
    window_size: int | None = None,
    stride: int | None = None,
) -> np.ndarray:
    if model_type == "1d":
        window_size = WINDOW_SIZE_1D if window_size is None else window_size
        stride = window_size if stride is None else stride
        return _window_signal(iq, window_size=window_size, stride=stride)

    window_size = WINDOW_SIZE_2D if window_size is None else window_size
    hop_length = window_size if stride is None else stride
    return _signal_to_spectrograms(iq, window_size=window_size, hop_length=hop_length)


def predict_processed(
    model: torch.nn.Module,
    processed: np.ndarray,
    class_names: list[str],
    device: torch.device,
    model_type: ModelType = "1d",
) -> dict[str, Any]:
    if model_type == "1d":
        window_probabilities, window_predictions = predict_windows_1d(
            model, processed, device=device
        )
        aggregated_probabilities = aggregate_probabilities(window_probabilities)
    else:
        window_probabilities, window_predictions = predict_windows_2d(
            model, processed, device=device
        )
        aggregated_probabilities = window_probabilities.mean(axis=0)

    final_index = int(np.argmax(aggregated_probabilities))
    final_class = class_names[final_index]
    final_confidence = float(aggregated_probabilities[final_index])

    window_results = []
    for window_number, (prediction_index, probabilities) in enumerate(
        zip(window_predictions, window_probabilities),
        start=1,
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
        "predicted_class": final_class,
        "confidence": final_confidence,
        "window_predictions": window_results,
        "class_probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(class_names, aggregated_probabilities)
        },
    }


def print_prediction(result: dict[str, Any]) -> None:
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {_format_percentage(result['confidence'])}")


def run_live_classifier(
    center_freq: float,
    model_type: ModelType = "1d",
    checkpoint_path: Path | str | None = None,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    gain: str | int = "auto",
    device_index: int = 0,
    window_size: int | None = None,
    stride: int | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = _load_class_names()

    if checkpoint_path is None:
        checkpoint_path = (
            BEST_MODEL_PATH_1D if model_type == "1d" else BEST_MODEL_PATH_2D
        )

    if model_type == "1d":
        model = load_trained_model_1d(
            device=device,
            checkpoint_path=checkpoint_path,
            num_classes=len(class_names),
        )
    else:
        model = load_trained_model_2d(
            device=device,
            checkpoint_path=checkpoint_path,
            num_classes=len(class_names),
        )

    sdr = RtlSdr(device_index=device_index)
    try:
        configure_sdr(
            sdr,
            center_freq=center_freq,
            sample_rate=sample_rate,
            gain=gain,
        )
        while True:
            iq = read_iq_samples(
                sdr,
                num_samples=num_samples,
            )
            processed = preprocess_iq(
                iq,
                model_type=model_type,
                window_size=window_size,
                stride=stride,
            )
            prediction = predict_processed(
                model,
                processed,
                class_names=class_names,
                device=device,
                model_type=model_type,
            )
            print_prediction(prediction)
    finally:
        sdr.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify live RTL-SDR IQ captures.")
    parser.add_argument("center_freq", type=float, help="Center frequency in Hz.")
    parser.add_argument(
        "--model",
        choices=("1d", "2d"),
        default="1d",
        help="Model pipeline to use.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of IQ samples per capture. Defaults to {DEFAULT_NUM_SAMPLES}.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Sample rate in Hz. Defaults to {DEFAULT_SAMPLE_RATE}.",
    )
    parser.add_argument(
        "--gain",
        default="auto",
        help='RF gain setting. Use "auto" or an integer gain value.',
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="RTL-SDR device index.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride / hop length. Defaults to the model window size.",
    )
    args = parser.parse_args()

    gain: str | int
    if isinstance(args.gain, str) and args.gain.isdigit():
        gain = int(args.gain)
    else:
        gain = args.gain

    try:
        run_live_classifier(
            center_freq=args.center_freq,
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            sample_rate=args.sample_rate,
            gain=gain,
            device_index=args.device_index,
            stride=args.stride,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
