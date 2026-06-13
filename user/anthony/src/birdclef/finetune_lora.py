"""LoRA finetuning for audio transformers on ESC-50."""

import argparse
import json
import random
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
    "beats": {
        # BEATs uses fairseq-style MultiheadAttention (q/k/v/out_proj). Checkpoint
        # is passed explicitly (BEATsEncoder(checkpoint_path=...)); there is no HF id.
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "embed_dim": 768,
        "default_lr": 1e-4,
    },
    "wav2vec2": {
        "model_name": "facebook/wav2vec2-base-960h",
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "embed_dim": 768,
        "default_lr": 1e-5,  # Lower LR to prevent gradient explosion
    },
    "ssamba-hf": {
        "checkpoint_path": "ssamba/ssamba_small_400.pth",  # Relative to checkpoint_dir
        "target_modules": ["in_proj", "out_proj"],  # Shared projections in Mamba blocks
        "embed_dim": 384,
        "default_lr": 1e-4,
        # ESC-50 normalization (matches original SSAMBA finetuning)
        "norm_mean": -6.6268077,
        "norm_std": 5.358466,
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


class SSAMBAHFForClassification(nn.Module):
    """SSAMBA-HF encoder with classification head.

    Matches the original SSAMBA finetuning setup:
    - LayerNorm -> Linear classification head
    - ESC-50 normalization stats
    - Noise augmentation + temporal roll
    - Optional overlapping patches with pos_embed interpolation
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        device: str = "cuda",
        noise_aug: bool = True,
        drop_path_rate: float = 0.1,
        target_length: int = 1024,
        fstride: int = 16,
        tstride: int = 16,
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
            checkpoint_path, device=device, verbose=True,
            use_cuda_kernels=True,
        )
        self.embed_dim = self.config.embed_dim  # 384 for small

        # Set stochastic depth across all layers (matching original SSAMBA: 0.1)
        self.encoder.set_drop_path_rate(drop_path_rate)

        # Adapt for finetuning with overlapping patches if needed
        self.target_length = target_length
        self._model_fdim = self.config.img_size[0]  # 128
        self._model_tdim = target_length
        if fstride != self.config.patch_size or tstride != self.config.patch_size:
            self.encoder.adapt_for_finetuning(
                finetune_img_size=(self._model_fdim, target_length),
                fstride=fstride,
                tstride=tstride,
            )
        elif target_length != self.config.img_size[1]:
            self.encoder.adapt_for_finetuning(
                finetune_img_size=(self._model_fdim, target_length),
                fstride=self.config.patch_size,
                tstride=self.config.patch_size,
            )

        # LayerNorm -> Linear head (matching original SSAMBA)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes),
        )

        # ESC-50 normalization
        ssamba_config = MODEL_CONFIGS["ssamba-hf"]
        self.norm_mean = ssamba_config["norm_mean"]
        self.norm_std = ssamba_config["norm_std"]

        # Noise augmentation
        self.noise_aug = noise_aug

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

            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
            fbanks.append(fbank)

        return torch.stack(fbanks, dim=0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass: waveform -> logits."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        fbank = self._waveform_to_fbank(waveform).to(self.device)

        # Noise augmentation during training (matching original SSAMBA dataloader.py:214-215)
        if self.noise_aug and self.training:
            fbank = fbank + torch.rand(fbank.shape[1], fbank.shape[2], device=fbank.device) * random.random() / 10
            fbank = torch.roll(fbank, random.randint(-10, 10), dims=1)

        embeddings = self.encoder.get_embeddings(fbank)
        return self.classifier(embeddings)


def create_model(
    encoder_type: str,
    num_classes: int,
    device: str,
    checkpoint_dir: Path = None,
    noise_aug: bool = True,
    drop_path_rate: float = 0.1,
    target_length: int = 1024,
    fstride: int = 16,
    tstride: int = 16,
) -> nn.Module:
    """Create model based on encoder type."""
    config = MODEL_CONFIGS[encoder_type]

    if encoder_type == "eat":
        return EATForClassification(config["model_name"], num_classes, device)
    elif encoder_type == "wav2vec2":
        return Wav2Vec2ForClassification(config["model_name"], num_classes, device)
    elif encoder_type == "ssamba-hf":
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required for ssamba-hf encoder")
        ckpt_path = checkpoint_dir / config["checkpoint_path"]
        return SSAMBAHFForClassification(
            str(ckpt_path), num_classes, device,
            noise_aug=noise_aug,
            drop_path_rate=drop_path_rate,
            target_length=target_length,
            fstride=fstride,
            tstride=tstride,
        )
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
    checkpoint_dir: Path = None,
    noise_aug: bool = True,
    drop_path_rate: float = 0.1,
    target_length: int = 1024,
    fstride: int = 16,
    tstride: int = 16,
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
        drop_last=True,
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
    model = create_model(
        encoder_type, num_classes=50, device=device,
        checkpoint_dir=checkpoint_dir,
        noise_aug=noise_aug,
        drop_path_rate=drop_path_rate,
        target_length=target_length,
        fstride=fstride,
        tstride=tstride,
    )
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
        choices=["eat", "wav2vec2", "ssamba-hf"],
        default="eat",
        help="Encoder type",
    )
    parser.add_argument("--esc50-root", type=str, required=True, help="ESC-50 dataset root")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (required for ssamba-hf)",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per fold")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: model-specific)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--fold", type=int, default=None, help="Single fold to run (1-5)")
    parser.add_argument("--no-noise-aug", action="store_true",
                        help="Disable noise augmentation (ssamba-hf only)")
    parser.add_argument("--drop-path-rate", type=float, default=0.1,
                        help="Drop path rate (ssamba-hf only)")
    parser.add_argument("--target-length", type=int, default=1024,
                        help="Target spectrogram time frames (ssamba-hf only)")
    parser.add_argument("--fstride", type=int, default=16,
                        help="Frequency stride for patch embedding (ssamba-hf only)")
    parser.add_argument("--tstride", type=int, default=16,
                        help="Time stride for patch embedding (ssamba-hf only)")
    args = parser.parse_args()

    # Use model-specific default LR if not specified
    if args.lr is None:
        args.lr = MODEL_CONFIGS[args.encoder]["default_lr"]
        print(f"Using default learning rate for {args.encoder}: {args.lr}")

    # Validate and parse checkpoint_dir
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if args.encoder == "ssamba-hf" and checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required for ssamba-hf encoder")

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
            lora_rank=args.lora_rank,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            max_grad_norm=args.max_grad_norm,
            checkpoint_dir=checkpoint_dir,
            noise_aug=not args.no_noise_aug,
            drop_path_rate=args.drop_path_rate,
            target_length=args.target_length,
            fstride=args.fstride,
            tstride=args.tstride,
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
    config = MODEL_CONFIGS[args.encoder]
    output = {
        "encoder": args.encoder,
        "model_name": config.get("model_name", config.get("checkpoint_path")),
        "lora_rank": args.lora_rank,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "batch_size": args.batch_size,
        "noise_aug": not args.no_noise_aug,
        "drop_path_rate": args.drop_path_rate,
        "target_length": args.target_length,
        "fstride": args.fstride,
        "tstride": args.tstride,
        "fold_accuracies": accuracies,
        "mean_accuracy": mean_acc,
    }

    fold_suffix = f"_fold{args.fold}" if args.fold else ""
    output_path = output_dir / f"{args.encoder}_lora_r{args.lora_rank}{fold_suffix}_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
