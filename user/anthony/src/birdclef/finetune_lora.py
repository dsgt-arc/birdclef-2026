"""LoRA finetuning for audio transformers on ESC-50."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from .data.esc50 import ESC50Dataset


# Model configurations
MODEL_CONFIGS = {
    "eat": {
        "model_name": "worstchan/EAT-base_epoch30_pretrain",
        "target_modules": ["qkv", "proj"],
        "embed_dim": 768,
        "default_lr": 1e-4,
    },
    "wav2vec2": {
        "model_name": "facebook/wav2vec2-base-960h",
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "embed_dim": 768,
        "default_lr": 1e-5,  # Lower LR to prevent gradient explosion
    },
}


class EATForClassification(nn.Module):
    """EAT encoder with classification head."""

    def __init__(self, model_name: str, num_classes: int, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel

        self.device = device
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.embed_dim = 768
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Mel spectrogram parameters
        self.target_length = 512
        self.norm_mean = -4.268
        self.norm_std = 4.569

    def _waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel-spectrogram."""
        import torchaudio

        batch_size = waveform.shape[0]
        mels = []

        for i in range(batch_size):
            wav = waveform[i]
            wav = wav - wav.mean()

            mel = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )

            n_frames = mel.shape[0]
            if n_frames < self.target_length:
                mel = torch.nn.functional.pad(
                    mel, (0, 0, 0, self.target_length - n_frames)
                )
            elif n_frames > self.target_length:
                mel = mel[: self.target_length, :]

            mels.append(mel)

        mel = torch.stack(mels, dim=0)
        mel = (mel - self.norm_mean) / (self.norm_std * 2)
        mel = mel.unsqueeze(1)  # (batch, 1, time, freq)

        return mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass: waveform -> logits."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        mel = self._waveform_to_mel(waveform).to(self.device)
        features = self.encoder.extract_features(mel)

        # Mean pool over sequence (excluding CLS token at position 0)
        pooled = features[:, 1:].mean(dim=1)

        return self.classifier(pooled)


class Wav2Vec2ForClassification(nn.Module):
    """Wav2Vec2 encoder with classification head."""

    def __init__(self, model_name: str, num_classes: int, device: str = "cuda"):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.device = device
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.embed_dim = self.encoder.config.hidden_size
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass: waveform -> logits."""
        # Remove channel dim if present: (batch, 1, samples) -> (batch, samples)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        waveform = waveform.to(self.device)

        # Wav2Vec2 expects (batch, samples) at 16kHz
        outputs = self.encoder(waveform)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, embed_dim)

        # Mean pool over sequence
        pooled = hidden_states.mean(dim=1)

        return self.classifier(pooled)


def create_model(encoder_type: str, num_classes: int, device: str) -> nn.Module:
    """Create model based on encoder type."""
    config = MODEL_CONFIGS[encoder_type]

    if encoder_type == "eat":
        return EATForClassification(config["model_name"], num_classes, device)
    elif encoder_type == "wav2vec2":
        return Wav2Vec2ForClassification(config["model_name"], num_classes, device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def apply_lora(model: nn.Module, encoder_type: str, rank: int = 8, alpha: int = 16) -> nn.Module:
    """Apply LoRA to the encoder."""
    config = MODEL_CONFIGS[encoder_type]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=config["target_modules"],
        bias="none",
    )

    # Apply LoRA to encoder only (not classifier)
    model.encoder = get_peft_model(model.encoder, lora_config)
    model.encoder.print_trainable_parameters()

    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
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

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> float:
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

    # Pad waveforms to same length
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
    lora_rank: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    max_grad_norm: float = 1.0,
) -> dict:
    """Run one fold of cross-validation."""
    print(f"\n=== Fold {fold}/5 ===")

    # Get fold splits
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

    # Create model with LoRA
    model = create_model(encoder_type, num_classes=50, device=device)
    model = apply_lora(model, encoder_type, rank=lora_rank)
    model = model.to(device)

    # Optimizer - only train LoRA params and classifier
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    print(f"Fold {fold} best accuracy: {best_acc:.4f}")

    return {"fold": fold, "best_acc": best_acc}


def main():
    parser = argparse.ArgumentParser(description="LoRA finetuning on ESC-50")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["eat", "wav2vec2"],
        default="eat",
        help="Encoder type",
    )
    parser.add_argument("--esc50-root", type=str, required=True, help="ESC-50 dataset root")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per fold")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: model-specific)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    args = parser.parse_args()

    # Use model-specific default LR if not specified
    if args.lr is None:
        args.lr = MODEL_CONFIGS[args.encoder]["default_lr"]
        print(f"Using default learning rate for {args.encoder}: {args.lr}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Encoder: {args.encoder}")
    print(f"LoRA rank: {args.lora_rank}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = ESC50Dataset(args.esc50_root, target_sr=16000)
    print(f"Dataset: {len(dataset)} samples, {dataset.num_classes} classes")

    # Run 5-fold CV
    results = []
    for fold in range(1, 6):
        fold_result = run_fold(
            fold=fold,
            dataset=dataset,
            encoder_type=args.encoder,
            lora_rank=args.lora_rank,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )
        results.append(fold_result)

    # Compute mean accuracy
    accuracies = [r["best_acc"] for r in results]
    mean_acc = sum(accuracies) / len(accuracies)

    print(f"\n=== Results ===")
    print(f"Encoder: {args.encoder}")
    print(f"Fold accuracies: {[f'{a:.4f}' for a in accuracies]}")
    print(f"Mean accuracy: {mean_acc:.4f}")

    # Save results
    output = {
        "encoder": args.encoder,
        "model_name": MODEL_CONFIGS[args.encoder]["model_name"],
        "lora_rank": args.lora_rank,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "batch_size": args.batch_size,
        "fold_accuracies": accuracies,
        "mean_accuracy": mean_acc,
    }

    output_path = output_dir / f"{args.encoder}_lora_r{args.lora_rank}_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
