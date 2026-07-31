from pathlib import Path
import numpy as np


WINDOW_SIZE = 2048


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


def iq_to_channels(sample):
    samples = np.asarray(sample)
    return np.stack((samples.real, samples.imag), axis=-1).astype(np.float32)


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


def summarize_dataset(X, y, batch_size=32, window_size=WINDOW_SIZE, hop_length=None):
    hop_length = window_size if hop_length is None else hop_length
    first_sample = _pad_to_window_multiple(normalize(X[0]), window_size, hop_length)
    first_window = iq_to_channels(first_sample[:window_size])
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

    return {
        "loaded_samples": len(X),
        "number_of_classes": len(np.unique(y)),
        "original_sample_shape": np.asarray(_extract_samples(X[0])).shape,
        "preprocessed_sample_shape": first_window.shape,
        "data_type": "torch.float32",
        "train": train_windows,
        "validation": val_windows,
        "test": test_windows,
        "batch_shape": f"torch.Size([{batch_size}, {first_window.shape[0]}, {first_window.shape[1]}])",
        "train_class_distribution": class_distribution,
    }


def create_dataloaders(
    X,
    y,
    batch_size=32,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    num_workers=0,
    window_size=WINDOW_SIZE,
    stride=None,
):
    import torch
    from torch.utils.data import DataLoader, Dataset

    stride = window_size if stride is None else stride

    class _WindowedIQDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = np.asarray(y, dtype=np.int64)
            self.window_size = window_size
            self.stride = stride
            self.window_lookup = []
            for sample_index, sample in enumerate(self.X):
                window_count = _window_count_after_padding(
                    _sample_length(sample), self.window_size, self.stride
                )
                for window_index in range(window_count):
                    self.window_lookup.append((sample_index, window_index))

        def __len__(self):
            return len(self.window_lookup)

        def __getitem__(self, index):
            sample_index, window_index = self.window_lookup[index]
            sample = _pad_to_window_multiple(
                normalize(self.X[sample_index]), self.window_size, self.stride
            )
            start = window_index * self.stride
            window = sample[start : start + self.window_size]
            return torch.from_numpy(iq_to_channels(window)), torch.tensor(
                self.y[sample_index], dtype=torch.long
            )

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = prepare_dataset(
        X,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    train_loader = DataLoader(
        _WindowedIQDataset(train_X, train_y),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        _WindowedIQDataset(val_X, val_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        _WindowedIQDataset(test_X, test_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
