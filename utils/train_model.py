import time
from typing import List, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm

from utils.metrics import calculate_metrics


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: str,
    print_every: int = 1,
    print_batch_loss: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Train a PyTorch model and save the best checkpoint based on validation F1.

    Notes:
        - Freezing/unfreezing of layers must be handled before constructing the optimizer.
        - The optimizer should be created from model parameters with requires_grad=True.

    Args:
        model: Model to train (with requires_grad already set).
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        criterion: Loss function.
        optimizer: Optimizer built on trainable params only.
        num_epochs: Number of epochs.
        device: CPU or CUDA device.
        save_path: Path to save best model (state_dict).
        print_every: Print epoch summaries every N epochs.
        print_batch_loss: Whether to print per-batch loss.

    Returns:
        (train_losses, val_losses): Lists with per-epoch losses.
    """
    # Basic sanity info
    print("\nModel device:", next(model.parameters()).device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    assert (
        trainable_params > 0
    ), "No trainable parameters found. Set requires_grad before calling train_model."

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_f1 = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # --- Training ---
        model.train()
        running_loss = 0.0

        train_loop = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Training",
            leave=False,
        )
        for batch_idx, (images, labels) in train_loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if print_batch_loss and (batch_idx + 1) % 10 == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds: List[Tensor] = []
        all_labels: List[Tensor] = []

        val_loop = tqdm(
            val_loader,
            total=len(val_loader),
            desc="Validating",
            leave=False,
        )
        with torch.no_grad():
            for images, labels in val_loop:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
                all_labels.append(labels)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        all_preds_tensor = torch.cat(all_preds)
        all_labels_tensor = torch.cat(all_labels)
        metrics = calculate_metrics(all_labels_tensor, all_preds_tensor)

        if (epoch + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(
                "Epoch [{}/{}] | Train Loss: {:.4f} | Val Loss: {:.4f} | "
                "Acc: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f} | "
                "Time: {:.1f}s".format(
                    epoch + 1,
                    num_epochs,
                    avg_train_loss,
                    avg_val_loss,
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    elapsed,
                )
            )

        # --- Save best by F1 ---
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved (F1: {best_f1:.4f}) → {save_path}")

    return train_losses, val_losses
