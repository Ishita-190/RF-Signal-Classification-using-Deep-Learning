from pathlib import Path
import json

import torch
import torch.nn as nn

from dataset import load_dataset, get_label_mapping
from model_2d.model_2d import SpectrogramCNN
from model_2d.preprocess_2d import create_dataloaders


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "2d"
BEST_MODEL_PATH = MODELS_DIR / "best_model_2d.pth"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"


def _run_epoch(model, loader, loss_fn, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    processed_batches = 0

    for inputs, targets in loader:
        if max_batches is not None and processed_batches >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
        processed_batches += 1

    mean_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return mean_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    num_epochs,
    device,
    max_batches_per_epoch=None,
    best_model_path=BEST_MODEL_PATH,
    training_history_path=TRAINING_HISTORY_PATH,
):
    best_val_accuracy = 0.0
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "best_validation_accuracy": 0.0,
    }

    best_model_path = Path(best_model_path)
    training_history_path = Path(training_history_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    training_history_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            max_batches=max_batches_per_epoch,
        )
        val_loss, val_accuracy = _run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            max_batches=max_batches_per_epoch,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs} — train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )
        epoch_path = MODELS_DIR / f"model_epoch_{epoch + 1}.pth"
        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, epoch_path)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    history["best_validation_accuracy"] = best_val_accuracy

    with training_history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    return model, history


def main(
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-3,
    max_batches_per_epoch=None,
    hop_length=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_dataset()
    train_loader, val_loader, _ = create_dataloaders(
        X, y, batch_size=batch_size, hop_length=hop_length
    )

    num_classes = len(get_label_mapping())
    model = SpectrogramCNN(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=num_epochs,
        device=device,
        max_batches_per_epoch=max_batches_per_epoch,
    )


if __name__ == "__main__":
    main()
