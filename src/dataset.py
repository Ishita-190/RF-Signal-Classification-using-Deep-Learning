from pathlib import Path

import numpy as np

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets_validated"


def get_label_mapping(dataset_dir=DATASET_DIR):
    dataset_dir = Path(dataset_dir)
    class_names = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    return {name: idx for idx, name in enumerate(class_names)}


def count_samples(dataset_dir=DATASET_DIR):
    dataset_dir = Path(dataset_dir)
    return sum(
        1
        for class_dir in dataset_dir.iterdir()
        if class_dir.is_dir()
        for _ in class_dir.glob("*.npy")
    )


def load_dataset(dataset_dir=DATASET_DIR):
    dataset_dir = Path(dataset_dir)
    class_to_idx = get_label_mapping(dataset_dir)
    X = []
    y = []
    for class_name in sorted(class_to_idx):
        for file_path in sorted((dataset_dir / class_name).glob("*.npy")):
            X.append(file_path)
            y.append(class_to_idx[class_name])
    return X, y
