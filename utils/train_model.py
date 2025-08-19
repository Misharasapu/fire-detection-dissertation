import time
import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    save_path,
    print_every=1,
    print_batch_loss=False
):
    """
    Train a PyTorch model and save the best checkpoint based on validation F1.

    Notes:
      - Freezing/unfreezing of layers must be handled in the notebook
        before constructing the optimizer.
      - The optimizer should be created from model parameters that have
        requires_grad=True.

    Args:
        model (nn.Module): Model to train (with requires_grad already set).
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer built on trainable params only.
        num_epochs (int): Number of epochs.
        device (torch.device): CPU or CUDA device.
        save_path (str): Path to save best model (state_dict).
        print_every (int): Print epoch summaries every N epochs.
        print_batch_loss (bool): Whether to print per-batch loss.

    Returns:
        (train_losses, val_losses): Lists with per-epoch losses.
    """

    print("\n🔍 Model device:", next(model.parameters()).device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Trainable parameters: {trainable_params}")
    assert trainable_params > 0, "No trainable parameters found. Set requires_grad before calling train_model."

    train_losses, val_losses = [], []
    best_f1 = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n🔁 Epoch {epoch + 1}/{num_epochs}")

        # --- Training ---
        model.train()
        running_loss = 0.0

        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), desc="🚂 Training", leave=False)
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
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        val_loop = tqdm(val_loader, total=len(val_loader), desc="🔎 Validating", leave=False)
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

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = calculate_metrics(all_labels, all_preds)

        if (epoch + 1) % print_every == 0:
            print(f"✅ Epoch [{epoch + 1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Acc: {metrics['accuracy']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"Time: {time.time() - start_time:.1f}s")

        # --- Save best by F1 ---
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), save_path)
            print(f"💾 New best model saved (F1: {best_f1:.4f}) → {save_path}")

    return train_losses, val_losses
