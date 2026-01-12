"""Knowledge distillation for sequence contextualization.

Teacher-Student distillation pipeline:
1. Teacher (BEATs Transformer) produces contextualized embeddings
2. Student (lightweight model) learns to predict teacher outputs from static embeddings

Reference: user/anthony/experiments/003-model-distillation/design.md
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add vendor path for BEATs imports
VENDOR_PATH = Path(__file__).parents[4] / "vendor" / "unilm" / "beats"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))


class DistillationLoss(nn.Module):
    """Combined MSE and Cosine Similarity loss for distillation.

    Args:
        mse_weight: Weight for MSE loss (default 0.5)
        cosine_weight: Weight for Cosine similarity loss (default 0.5)
    """

    def __init__(self, mse_weight: float = 0.5, cosine_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(
        self, student_embeds: torch.Tensor, teacher_embeds: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            student_embeds: (batch, seq_len, embed_dim)
            teacher_embeds: (batch, seq_len, embed_dim)

        Returns:
            Dictionary with 'loss', 'mse', 'cosine' tensors
        """
        # MSE loss
        mse = F.mse_loss(student_embeds, teacher_embeds)

        # Cosine similarity loss (per-patch)
        cos_sim = F.cosine_similarity(student_embeds, teacher_embeds, dim=-1)
        cosine_loss = 1 - cos_sim.mean()

        # Combined loss
        loss = self.mse_weight * mse + self.cosine_weight * cosine_loss

        return {
            "loss": loss,
            "mse": mse,
            "cosine": cosine_loss,
            "cosine_sim": cos_sim.mean(),
        }


class DistillationTrainer:
    """Trainer for student model distillation.

    Handles:
    - Teacher inference (BEATs transformer)
    - Student training loop
    - Logging and checkpointing

    Args:
        student: Student model to train
        teacher_encoder: BEATs encoder model
        tokenizer: BEATs tokenizer for static embedding lookup
        static_embeddings: Pre-computed static embeddings (num_tokens, embed_dim)
        device: Device to train on
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        mse_weight: MSE loss weight
        cosine_weight: Cosine loss weight
    """

    def __init__(
        self,
        student: nn.Module,
        teacher_encoder: nn.Module,
        tokenizer: nn.Module,
        static_embeddings: torch.Tensor,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        mse_weight: float = 0.5,
        cosine_weight: float = 0.5,
    ):
        self.student = student.to(device)
        self.teacher = teacher_encoder.to(device)
        self.tokenizer = tokenizer.to(device)
        self.static_embeddings = static_embeddings.to(device)
        self.device = device

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Freeze tokenizer
        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Loss
        self.criterion = DistillationLoss(mse_weight, cosine_weight)

        # Tracking
        self.step = 0
        self.epoch = 0

    def get_teacher_embeddings(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Get contextualized embeddings from teacher.

        Args:
            waveforms: (batch, samples) at 16kHz

        Returns:
            Teacher embeddings (batch, seq_len, embed_dim)
        """
        with torch.no_grad():
            # BEATs extract_features returns (embeddings, padding_mask)
            embeddings, _ = self.teacher.extract_features(waveforms, padding_mask=None)
        return embeddings

    def get_student_inputs(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Get static embeddings for student input.

        Args:
            waveforms: (batch, samples) at 16kHz

        Returns:
            Static embeddings (batch, seq_len, embed_dim)
        """
        with torch.no_grad():
            # Get token IDs from tokenizer
            token_ids = self.tokenizer.extract_labels(waveforms)  # (batch, seq_len)
            # Lookup static embeddings
            static_embeds = self.static_embeddings[token_ids]  # (batch, seq_len, embed_dim)
        return static_embeds

    def train_step(self, waveforms: torch.Tensor) -> dict[str, float]:
        """Single training step.

        Args:
            waveforms: (batch, samples) at 16kHz

        Returns:
            Dictionary of loss values
        """
        self.student.train()
        waveforms = waveforms.to(self.device)

        # Get teacher targets
        teacher_embeds = self.get_teacher_embeddings(waveforms)

        # Get student inputs and forward pass
        student_inputs = self.get_student_inputs(waveforms)
        student_embeds = self.student(student_inputs)

        # Compute loss
        loss_dict = self.criterion(student_embeds, teacher_embeds)
        loss = loss_dict["loss"]

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1

        return {k: v.item() for k, v in loss_dict.items()}

    def train_epoch(
        self,
        dataloader: DataLoader,
        max_steps: Optional[int] = None,
        log_interval: int = 100,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader yielding (waveform, ...) tuples
            max_steps: Maximum steps per epoch (None for full epoch)
            log_interval: Steps between logging

        Returns:
            Average loss values for the epoch
        """
        total_loss = 0.0
        total_mse = 0.0
        total_cosine = 0.0
        n_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                waveforms = batch[0]
            else:
                waveforms = batch

            # Ensure correct shape
            if waveforms.dim() == 3:
                waveforms = waveforms.squeeze(1)

            loss_dict = self.train_step(waveforms)

            total_loss += loss_dict["loss"]
            total_mse += loss_dict["mse"]
            total_cosine += loss_dict["cosine"]
            n_steps += 1

            if self.step % log_interval == 0:
                pbar.set_postfix(
                    loss=f"{loss_dict['loss']:.4f}",
                    mse=f"{loss_dict['mse']:.4f}",
                    cos=f"{loss_dict['cosine_sim']:.4f}",
                )

            if max_steps and n_steps >= max_steps:
                break

        self.epoch += 1

        return {
            "loss": total_loss / n_steps,
            "mse": total_mse / n_steps,
            "cosine": total_cosine / n_steps,
        }

    def save_checkpoint(self, path: str | Path):
        """Save training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "epoch": self.epoch,
                "student_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def load_teacher_and_tokenizer(
    encoder_path: str | Path,
    tokenizer_path: str | Path,
) -> tuple[nn.Module, nn.Module]:
    """Load BEATs teacher encoder and tokenizer.

    Args:
        encoder_path: Path to BEATs_iter3.pt
        tokenizer_path: Path to Tokenizers_iter3.pt

    Returns:
        Tuple of (encoder, tokenizer)
    """
    from BEATs import BEATs, BEATsConfig
    from Tokenizers import Tokenizers, TokenizersConfig

    # Load encoder
    encoder_ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    encoder_cfg = BEATsConfig(encoder_ckpt["cfg"])
    encoder = BEATs(encoder_cfg)
    encoder.load_state_dict(encoder_ckpt["model"])
    encoder.eval()

    # Load tokenizer
    tokenizer_ckpt = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
    tokenizer_cfg = TokenizersConfig(tokenizer_ckpt["cfg"])
    tokenizer = Tokenizers(tokenizer_cfg)
    tokenizer.load_state_dict(tokenizer_ckpt["model"])
    tokenizer.eval()

    return encoder, tokenizer
