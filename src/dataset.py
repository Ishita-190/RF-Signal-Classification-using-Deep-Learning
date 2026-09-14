from pathlib import Path

import numpy as np

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets_validated"
HF_REPO_ID = "ishisan28/rf-signal-dataset"


CLASSES = ["ADS_B", "FM_broadcast", "ISM_sensors", "noise"]


def download_dataset(dest=DATASET_DIR, repo_id=HF_REPO_ID):
    from huggingface_hub import snapshot_download

    dest = Path(dest).resolve()
    if not str(dest).startswith(str(Path(__file__).resolve().parents[1])):
        raise ValueError(f"Destination path {dest} is outside the project directory.")

    if all((dest / cls).exists() and any((dest / cls).glob("*.npy")) for cls in CLASSES):
        print(f"Dataset already exists at {dest}, skipping download.")
        return dest

    print(f"Downloading dataset from {repo_id} to {dest}...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dest))
    print("Download complete.")
    return dest


def get_label_mapping(dataset_dir=DATASET_DIR):
    dataset_dir = Path(dataset_dir)
    class_names = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    return {name: idx for idx, name in enumerate(class_names)}


def count_samples(dataset_dir=DATASET_DIR):
    dataset_dir = Path(dataset_dir)
    return sum(
        1
        for class_dir in dataset_dir.iterdir()
        if class_dir.is_dir() and not class_dir.name.startswith(".")
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
