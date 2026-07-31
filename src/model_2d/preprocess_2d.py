from __future__ import annotations
import sys

import numpy as np
from scipy import ndimage, signal
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


WINDOW_SIZE = 2048
SPECTROGRAM_SIZE = (128, 128)


def _extract_samples(sample):
    if isinstance(sample, Path):
        sample = np.load(sample, allow_pickle=True)
    if isinstance(sample, np.ndarray) and sample.dtype == object:
        sample = sample.item()
    if isinstance(sample, dict):
        return sample["samples"]
    return sample


def _sample_length(sample):
    return np.asarray(_extract_samples(sample)).reshape(-1).size


def normalize(sample):
    samples = np.asarray(_extract_samples(sample), dtype=np.complex64)
    max_amplitude = np.abs(samples).max()
    if max_amplitude == 0:
        return samples
    return samples / max_amplitude


def _pad_to_window_multiple(sample, window_size, hop_length):
    sample = np.asarray(sample, dtype=np.complex64)
    if sample.ndim != 1:
        sample = sample.reshape(-1)

    if sample.size <= window_size:
        pad_length = window_size - sample.size
    else:
        remainder = (sample.size - window_size) % hop_length
        pad_length = 0 if remainder == 0 else hop_length - remainder

    if pad_length > 0:
        sample = np.pad(sample, (0, pad_length), mode="constant")

    return sample


def _window_count_after_padding(sample_length, window_size, hop_length):
    if sample_length <= window_size:
        return 1
    remainder = (sample_length - window_size) % hop_length
    pad_length = 0 if remainder == 0 else hop_length - remainder
    padded_length = sample_length + pad_length
    return 1 + (padded_length - window_size) // hop_length


def _resize_spectrogram(spectrogram, target_size=SPECTROGRAM_SIZE):
    spectrogram = np.asarray(spectrogram, dtype=np.float32)
    if spectrogram.ndim != 2:
        raise ValueError(f"Expected a 2D spectrogram, got shape {spectrogram.shape}")

    target_height, target_width = target_size
    zoom_factors = (
        target_height / spectrogram.shape[0],
        target_width / spectrogram.shape[1],
    )
    resized = ndimage.zoom(spectrogram, zoom_factors, order=1)
    return resized.astype(np.float32)


def window_to_spectrogram(window, target_size=SPECTROGRAM_SIZE):
    window = np.asarray(window, dtype=np.complex64)
    if window.ndim != 1:
        window = window.reshape(-1)

    if window.size == 0:
        raise ValueError("Cannot generate a spectrogram from an empty window")

    nperseg = min(256, window.size)
    noverlap = min(nperseg - 1, nperseg // 2) if nperseg > 1 else 0
    _, _, power = signal.spectrogram(
        window,
        fs=1.0,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=False,
        mode="magnitude",
    )
    power = np.log1p(power).astype(np.float32)
    power -= power.min()
    max_value = power.max()
    if max_value > 0:
        power /= max_value
    return _resize_spectrogram(power, target_size=target_size)


def save_example_spectrograms(
    X,
    y,
    OUTPUT_DIR,
    num_examples=None,
    window_size=WINDOW_SIZE,
    label_names=None,
):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_arr = np.asarray(y)
    if num_examples is None:
        seen = set()
        indices = []
        for i, label in enumerate(y_arr):
            if label not in seen:
                seen.add(label)
                indices.append(i)
    else:
        indices = list(range(min(num_examples, len(X))))

    for plot_idx, i in enumerate(indices):
        sample = normalize(X[i])
        sample = _pad_to_window_multiple(sample, window_size, window_size)

        window = sample[:window_size]
        spectrogram = window_to_spectrogram(window)

        plt.figure(figsize=(5, 5))
        plt.imshow(spectrogram, origin="lower", aspect="auto", cmap="viridis")
        plt.colorbar(label="Normalized Magnitude")
        plt.title(label_names[y[i]] if label_names else f"Label {y[i]}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(output_dir / f"spectrogram_{plot_idx}.png")
        plt.close()

    return output_dir


def prepare_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    indices = np.arange(len(X))
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    train_idx = []
    val_idx = []
    test_idx = []
    for label in np.unique(y):
        label_indices = indices[y == label]
        rng.shuffle(label_indices)
        n_total = len(label_indices)
        n_test = int(round(n_total * test_size))
        n_val = int(round(n_total * val_size))
        n_train = n_total - n_val - n_test
        train_idx.extend(label_indices[:n_train])
        val_idx.extend(label_indices[n_train : n_train + n_val])
        test_idx.extend(label_indices[n_train + n_val :])
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    return (
        ([X[i] for i in train_idx], y[train_idx]),
        ([X[i] for i in val_idx], y[val_idx]),
        ([X[i] for i in test_idx], y[test_idx]),
    )


def summarize_dataset(
    X,
    y,
    batch_size=8,
    window_size=WINDOW_SIZE,
    hop_length=None,
    spectrogram_size=SPECTROGRAM_SIZE,
    output_dir=None,
    label_names=None,
):
    hop_length = window_size if hop_length is None else hop_length
    first_sample = _pad_to_window_multiple(normalize(X[0]), window_size, hop_length)
    first_window = first_sample[:window_size]
    first_spectrogram = window_to_spectrogram(
        first_window, target_size=spectrogram_size
    )
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = prepare_dataset(X, y)

    train_windows = 0
    val_windows = 0
    test_windows = 0
    class_distribution = {}

    for sample, label in zip(train_X, train_y):
        window_count = _window_count_after_padding(
            _sample_length(sample), window_size, hop_length
        )
        train_windows += window_count
        class_distribution[int(label)] = (
            class_distribution.get(int(label), 0) + window_count
        )

    for sample in val_X:
        val_windows += _window_count_after_padding(
            _sample_length(sample), window_size, hop_length
        )

    for sample in test_X:
        test_windows += _window_count_after_padding(
            _sample_length(sample), window_size, hop_length
        )

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parents[2] / "results" / "2d" / "spectrograms"
        )
    output_dir = Path(output_dir)
    save_example_spectrograms(
        X, y, output_dir, window_size=window_size, label_names=label_names
    )

    return {
        "loaded_samples": len(X),
        "number_of_classes": len(np.unique(y)),
        "original_sample_shape": np.asarray(_extract_samples(X[0])).shape,
        "preprocessed_sample_shape": (1, *first_spectrogram.shape),
        "data_type": "torch.float32",
        "train": train_windows,
        "validation": val_windows,
        "test": test_windows,
        "batch_shape": f"torch.Size([{batch_size}, 1, {first_spectrogram.shape[0]}, {first_spectrogram.shape[1]}])",
        "train_class_distribution": class_distribution,
        "example_spectrograms_dir": str(output_dir),
    }


def create_dataloaders(
    X,
    y,
    batch_size=8,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    num_workers=0,
    window_size=WINDOW_SIZE,
    hop_length=None,
    spectrogram_size=SPECTROGRAM_SIZE,
):

    hop_length = window_size if hop_length is None else hop_length

    class _SpectrogramIQDataset(Dataset):
        def __init__(self, X, y):
            print("Indexing windows...")

            self.samples = []
            self.windows = []

            y_arr = np.asarray(y, dtype=np.int64)

            for sample, label in zip(X, y_arr):
                padded = _pad_to_window_multiple(
                    normalize(sample),
                    window_size,
                    hop_length,
                )

                sample_idx = len(self.samples)
                self.samples.append(padded)

                window_count = _window_count_after_padding(
                    padded.size,
                    window_size,
                    hop_length,
                )

                for i in range(window_count):
                    start = i * hop_length
                    self.windows.append((sample_idx, start, label))

            print(f"Indexed {len(self.windows)} windows.")

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, index):
            sample_idx, start, label = self.windows[index]

            sample = self.samples[sample_idx]

            window = sample[start : start + window_size]

            spectrogram = window_to_spectrogram(
                window,
                target_size=spectrogram_size,
            )

            spectrogram = spectrogram[np.newaxis, :, :]

            return (
                torch.from_numpy(spectrogram),
                torch.tensor(label, dtype=torch.long),
            )

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = prepare_dataset(
        X,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    train_loader = DataLoader(
        _SpectrogramIQDataset(train_X, train_y),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        _SpectrogramIQDataset(val_X, val_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        _SpectrogramIQDataset(test_X, test_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
