"""Full finetuning for audio transformers on ESC-50.

Differences from finetune_lora.py:
- Full finetuning (all parameters trainable)
- SpecAugment data augmentation (frequency/time masking)
- ESC-50 specific normalization constants
- Single fold option for quick testing
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.esc50 import ESC50Dataset


# Model configurations
MODEL_CONFIGS = {
    "ssamba-hf": {
        "checkpoint_path": "ssamba/ssamba_small_400.pth",
        "embed_dim": 384,
        "default_lr": 1e-4,
        # AudioSet normalization (matches pre-trained weights)
        "norm_mean": -4.2677393,
        "norm_std": 4.5689974,
    },
}


class SpecAugment(nn.Module):
    """SpecAugment: frequency and time masking for spectrograms."""

    def __init__(self, freq_mask_param: int = 24, time_mask_param: int = 96):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram.

        Args:
            x: (batch, time, freq) spectrogram

        Returns:
            Augmented spectrogram
        """
        if not self.training:
            return x

        batch_size, time_dim, freq_dim = x.shape

        # Frequency masking
        if self.freq_mask_param > 0:
            for i in range(batch_size):
                f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
                f0 = torch.randint(0, max(1, freq_dim - f), (1,)).item()
                x[i, :, f0 : f0 + f] = 0

        # Time masking
        if self.time_mask_param > 0:
            for i in range(batch_size):
                t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
                t0 = torch.randint(0, max(1, time_dim - t), (1,)).item()
                x[i, t0 : t0 + t, :] = 0

        return x


class SSAMBAHFForClassification(nn.Module):
    """SSAMBA-HF encoder with classification head for full finetuning."""

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        device: str = "cuda",
        use_specaugment: bool = True,
        freq_mask: int = 24,
        time_mask: int = 96,
    ):
        super().__init__()
        import sys
        from pathlib import Path

        # Add ssamba_hf to path
        ssamba_hf_dir = (
            Path(__file__).parents[2]
            / "experiments"
            / "004-ssamba-hf-port"
            / "ssamba_hf"
        )
        if str(ssamba_hf_dir) not in sys.path:
            sys.path.insert(0, str(ssamba_hf_dir))

        from ssamba_hf import load_from_ssamba_checkpoint

        self.device = device
        self.encoder, self.config = load_from_ssamba_checkpoint(
            checkpoint_path,
            device=device,
            verbose=True,
            use_cuda_kernels=True,
        )
        self.embed_dim = self.config.embed_dim
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Preprocessing params
        self._model_fdim, self._model_tdim = self.config.img_size  # (128, 1024)
        # AudioSet normalization (matches pre-trained weights)
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

        # SpecAugment
        self.use_specaugment = use_specaugment
        if use_specaugment:
            self.spec_augment = SpecAugment(freq_mask, time_mask)
            print(f"SpecAugment enabled: freq_mask={freq_mask}, time_mask={time_mask}")

    def _waveform_to_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to filter bank features."""
        import torchaudio

        batch_size = waveform.shape[0]
        fbanks = []

        target_tdim = self._model_tdim
        target_fdim = self._model_fdim

        for i in range(batch_size):
            wav = waveform[i]
            wav = wav - wav.mean()

            fbank = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=target_fdim,
                dither=0.0,
                frame_shift=10,
            )

            n_frames = fbank.shape[0]
            if n_frames < target_tdim:
                fbank = torch.nn.functional.pad(
                    fbank, (0, 0, 0, target_tdim - n_frames)
                )
            elif n_frames > target_tdim:
                fbank = fbank[:target_tdim, :]

            # Normalize with ESC-50 specific values
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
            fbanks.append(fbank)

        return torch.stack(fbanks, dim=0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass: waveform -> logits."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        fbank = self._waveform_to_fbank(waveform).to(self.device)

        # Apply SpecAugment during training
        if self.use_specaugment and self.training:
            fbank = self.spec_augment(fbank)

        embeddings = self.encoder.get_embeddings(fbank)
        return self.classifier(embeddings)


def create_model(
    encoder_type: str,
    num_classes: int,
    device: str,
    checkpoint_dir: Path,
    use_specaugment: bool = True,
    freq_mask: int = 24,
    time_mask: int = 96,
) -> nn.Module:
    """Create model for full finetuning."""
    config = MODEL_CONFIGS[encoder_type]

    if encoder_type == "ssamba-hf":
        ckpt_path = checkpoint_dir / config["checkpoint_path"]
        return SSAMBAHFForClassification(
            str(ckpt_path),
            num_classes,
            device,
            use_specaugment=use_specaugment,
            freq_mask=freq_mask,
            time_mask=time_mask,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: str,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for waveforms, labels in tqdm(train_loader, desc="Training", leave=False):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: str) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    for waveforms, labels in tqdm(val_loader, desc="Evaluating", leave=False):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        logits = model(waveforms)
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def collate_fn(batch):
    """Collate function for DataLoader."""
    waveforms, labels = zip(*batch)

    max_len = max(w.shape[-1] for w in waveforms)
    padded = []
    for w in waveforms:
        if w.shape[-1] < max_len:
            pad_amount = max_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad_amount))
        padded.append(w)

    waveforms = torch.stack(padded, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return waveforms, labels


def run_fold(
    fold: int,
    dataset: ESC50Dataset,
    encoder_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    checkpoint_dir: Path,
    max_grad_norm: float = 1.0,
    use_specaugment: bool = True,
    freq_mask: int = 24,
    time_mask: int = 96,
    lr_decay: float = 0.85,
    warmup_epochs: int = 5,
) -> dict:
    """Run one fold of cross-validation with full finetuning."""
    print(f"\n=== Fold {fold}/5 ===")

    train_dataset, val_dataset = dataset.get_fold_splits(fold)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Create model - all parameters trainable
    model = create_model(
        encoder_type,
        num_classes=50,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_specaugment=use_specaugment,
        freq_mask=freq_mask,
        time_mask=time_mask,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler: decay after warmup
    # Similar to SSAMBA: decay by 0.85 every epoch after warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 1.0
        else:
            return lr_decay ** (epoch - warmup_epochs)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, None, criterion, device, max_grad_norm
        )
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        if val_acc > best_acc:
            best_acc = val_acc

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        # Log every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, "
                f"val_acc={val_acc:.4f}, lr={current_lr:.2e}"
            )

    print(f"Fold {fold} best accuracy: {best_acc:.4f}")

    return {"fold": fold, "best_acc": best_acc, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Full finetuning on ESC-50")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["ssamba-hf"],
        default="ssamba-hf",
        help="Encoder type",
    )
    parser.add_argument("--esc50-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fold", type=int, default=None, help="Single fold to run (1-5)")
    parser.add_argument("--no-specaugment", action="store_true")
    parser.add_argument("--freq-mask", type=int, default=24)
    parser.add_argument("--time-mask", type=int, default=96)
    parser.add_argument("--lr-decay", type=float, default=0.85)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}")
    print(f"Full finetuning (all parameters)")
    print(f"SpecAugment: {not args.no_specaugment}")
    print(f"LR: {args.lr}, decay: {args.lr_decay} after {args.warmup_epochs} epochs")

    dataset = ESC50Dataset(args.esc50_root, target_sr=16000)
    print(f"Dataset: {len(dataset)} samples, {dataset.num_classes} classes")

    # Run folds
    if args.fold is not None:
        folds = [args.fold]
        print(f"Running single fold: {args.fold}")
    else:
        folds = list(range(1, 6))

    results = []
    for fold in folds:
        fold_result = run_fold(
            fold=fold,
            dataset=dataset,
            encoder_type=args.encoder,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            checkpoint_dir=checkpoint_dir,
            max_grad_norm=args.max_grad_norm,
            use_specaugment=not args.no_specaugment,
            freq_mask=args.freq_mask,
            time_mask=args.time_mask,
            lr_decay=args.lr_decay,
            warmup_epochs=args.warmup_epochs,
        )
        results.append(fold_result)

    # Summary
    accuracies = [r["best_acc"] for r in results]
    mean_acc = sum(accuracies) / len(accuracies)

    print(f"\n=== Results ===")
    print(f"Encoder: {args.encoder}")
    print(f"Fold accuracies: {[f'{a:.4f}' for a in accuracies]}")
    print(f"Mean accuracy: {mean_acc:.4f}")

    # Save results
    output = {
        "encoder": args.encoder,
        "method": "full_finetune",
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "specaugment": not args.no_specaugment,
        "freq_mask": args.freq_mask,
        "time_mask": args.time_mask,
        "fold_results": results,
        "mean_accuracy": mean_acc,
    }

    fold_suffix = f"_fold{args.fold}" if args.fold else ""
    output_path = output_dir / f"{args.encoder}_full_finetune{fold_suffix}_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
