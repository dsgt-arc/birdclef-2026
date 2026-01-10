"""Linear probe training for audio classification."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .encoders.base import AudioEncoder


class LinearProbe(nn.Module):
    """Linear classifier on top of frozen encoder features."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def extract_features_from_dataloader(
    encoder: AudioEncoder,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features for entire dataset using frozen encoder.

    Args:
        encoder: Audio encoder model
        dataloader: DataLoader for the dataset
        device: Device to use

    Returns:
        Tuple of (features, labels) tensors
    """
    all_features = []
    all_labels = []

    encoder.eval()
    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Extracting features"):
            waveforms = waveforms.to(device)
            features = encoder.extract_features(waveforms)
            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
) -> tuple[LinearProbe, dict]:
    """Train a linear probe on extracted features.

    Args:
        train_features: Training features (N, embed_dim)
        train_labels: Training labels (N,)
        val_features: Validation features (M, embed_dim)
        val_labels: Validation labels (M,)
        num_classes: Number of classes
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to use

    Returns:
        Trained linear probe and training history
    """
    input_dim = train_features.shape[1]
    probe = LinearProbe(input_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create simple batching
    n_train = len(train_features)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        probe.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            batch_features = train_features[idx]
            batch_labels = train_labels[idx]

            optimizer.zero_grad()
            logits = probe(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_train_loss)

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_features)
            val_loss = criterion(val_logits, val_labels).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = probe.state_dict().copy()

    # Load best model
    if best_state is not None:
        probe.load_state_dict(best_state)

    return probe, history


def evaluate_5fold_cv(
    encoder: AudioEncoder,
    dataset,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
) -> dict:
    """Evaluate encoder using 5-fold cross-validation.

    Args:
        encoder: Audio encoder model
        dataset: ESC50Dataset (full dataset)
        batch_size: Batch size for feature extraction
        epochs: Training epochs for linear probe
        lr: Learning rate
        device: Device to use

    Returns:
        Dictionary with fold accuracies and mean accuracy
    """
    fold_accs = []

    for val_fold in range(1, 6):
        print(f"\n=== Fold {val_fold}/5 ===")

        train_ds, val_ds = dataset.get_fold_splits(val_fold)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Extract features
        print("Extracting training features...")
        train_features, train_labels = extract_features_from_dataloader(
            encoder, train_loader, device
        )
        print("Extracting validation features...")
        val_features, val_labels = extract_features_from_dataloader(
            encoder, val_loader, device
        )

        # Train linear probe
        print("Training linear probe...")
        probe, history = train_linear_probe(
            train_features,
            train_labels,
            val_features,
            val_labels,
            num_classes=dataset.num_classes,
            epochs=epochs,
            lr=lr,
            device=device,
        )

        # Final evaluation
        probe.eval()
        with torch.no_grad():
            val_features = val_features.to(device)
            val_logits = probe(val_features)
            val_preds = val_logits.argmax(dim=1)
            val_labels = val_labels.to(device)
            acc = (val_preds == val_labels).float().mean().item()

        fold_accs.append(acc)
        print(f"Fold {val_fold} accuracy: {acc:.4f}")

    mean_acc = sum(fold_accs) / len(fold_accs)
    print(f"\n=== Mean accuracy: {mean_acc:.4f} ===")

    return {
        "fold_accuracies": fold_accs,
        "mean_accuracy": mean_acc,
    }
