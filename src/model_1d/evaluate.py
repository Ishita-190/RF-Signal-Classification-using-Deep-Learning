from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import get_label_mapping, load_dataset
from model_1d.model import SignalCNN
from model_1d.preprocess import create_dataloaders


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "1d"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrix"
BEST_MODEL_PATH = MODELS_DIR / "best_model_1d.pth"
EVALUATION_PATH = RESULTS_DIR / "evaluation.json"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_IMAGE_PATH = CONFUSION_MATRIX_DIR / "confusion_matrix.png"


def load_trained_model(device, checkpoint_path=BEST_MODEL_PATH, num_classes=5):
    model = SignalCNN(num_classes=num_classes).to(device)
    checkpoint_path = Path(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model, test_loader, loss_fn, device):
    all_targets = []
    all_predictions = []
    total_loss = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            predictions = torch.argmax(logits, dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    test_loss = total_loss / max(total_samples, 1)
    test_accuracy = accuracy_score(all_targets, all_predictions)
    macro_precision = precision_score(
        all_targets, all_predictions, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        all_targets, all_predictions, average="macro", zero_division=0
    )
    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    weighted_precision = precision_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )
    weighted_recall = recall_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )
    weighted_f1 = f1_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "y_true": all_targets,
        "y_pred": all_predictions,
    }


def save_classification_report(y_true, y_pred, class_names, labels, output_path):
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def plot_confusion_matrix(y_true, y_pred, class_names, labels, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            value = cm[row_index, column_index]
            color = "white" if value > threshold else "black"
            plt.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=color,
            )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return cm


def evaluate(
    checkpoint_path=BEST_MODEL_PATH, batch_size=32, stride=2048, num_workers=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_dataset()
    _, _, test_loader = create_dataloaders(
        X, y, batch_size=batch_size, num_workers=num_workers, stride=stride
    )
    label_mapping = get_label_mapping()
    class_names = [
        name for name, _ in sorted(label_mapping.items(), key=lambda item: item[1])
    ]
    labels = list(range(len(class_names)))

    model = load_trained_model(
        device, checkpoint_path=checkpoint_path, num_classes=len(class_names)
    )
    loss_fn = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, loss_fn, device)

    cm = plot_confusion_matrix(
        metrics["y_true"],
        metrics["y_pred"],
        class_names,
        labels,
        CONFUSION_MATRIX_IMAGE_PATH,
    )
    report_text = save_classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        class_names,
        labels,
        CLASSIFICATION_REPORT_PATH,
    )

    evaluation_payload = {
        "test_loss": metrics["test_loss"],
        "test_accuracy": metrics["test_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with EVALUATION_PATH.open("w", encoding="utf-8") as evaluation_file:
        json.dump(evaluation_payload, evaluation_file, indent=2)

    return evaluation_payload


if __name__ == "__main__":
    evaluate()
