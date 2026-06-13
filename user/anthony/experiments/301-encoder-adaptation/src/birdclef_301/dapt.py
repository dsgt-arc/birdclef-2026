"""Soundscape DAPT — domain-adaptive continued pretraining (EAT / SSAMBA-HF).

A data2vec / BYOL-style latent-bootstrap objective on the unlabeled Pantanal
soundscape pool: a masked student predicts the full-clip embedding of an EMA
teacher. This is encoder-agnostic (it only needs the wrapper's pooled forward), so
EAT and SSAMBA-HF share one loop. It is a generic continued-pretraining objective
rather than each encoder's exact original SSL loss — the genericity tradeoff is
deliberate and noted in the proposal. (SSAMBA's exact masked-reconstruction DAPT
can alternatively reuse exp 103's pipeline.)

BEATs is excluded by design — its masked-prediction SSL needs a co-trained acoustic
tokenizer (see proposal). DAPT updates full encoder weights and saves a state_dict
that exp 300 loads via `birdclef-300 extract --variant dapt-... --state-dict <pt>`.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from birdclef_300.registry import build_torch_encoder
from birdclef_301.data import SoundscapeDataset
from birdclef_301.encoder_forward import pooled_features
from birdclef_301.losses import latent_bootstrap_loss

logger = logging.getLogger("birdclef_301.dapt")


def _time_mask(wav: torch.Tensor, frac: float, n_spans: int, rng: torch.Generator) -> torch.Tensor:
    """Zero `n_spans` random time spans covering ~`frac` of each (1, N) waveform."""
    out = wav.clone()
    n = wav.shape[-1]
    span = max(1, int(frac * n / max(1, n_spans)))
    starts = torch.randint(0, max(1, n - span), (wav.shape[0], n_spans), generator=rng)
    for b, row in enumerate(starts.tolist()):
        for start in row:
            out[b, ..., start:start + span] = 0.0
    return out


@torch.no_grad()
def _ema_update(teacher, student, momentum: float) -> None:
    for tp, sp in zip(teacher.model.parameters(), student.model.parameters()):
        tp.mul_(momentum).add_(sp, alpha=1.0 - momentum)
    for tb, sb in zip(teacher.model.buffers(), student.model.buffers()):
        tb.copy_(sb)


def run_dapt(cfg: dict, output_dir: str | Path, max_steps: int | None = None) -> dict:
    encoder = cfg["encoder"]
    if encoder not in ("eat", "ssamba_hf"):
        raise ValueError(f"DAPT roster is eat/ssamba_hf; got '{encoder}' (BEATs is LoRA-only)")
    enc_cfg = cfg["encoder_cfg"]
    dapt_cfg = cfg.get("dapt", {})
    train_cfg = cfg.get("training", {})
    device = enc_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; using CPU")
        device = "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = SoundscapeDataset(cfg["data"]["soundscape_dir"])
    loader = DataLoader(
        ds, batch_size=int(train_cfg.get("batch_size", 16)), shuffle=True,
        drop_last=True, num_workers=int(train_cfg.get("num_workers", 4)),
    )
    logger.info("Soundscape DAPT pool: %d files", len(ds))

    student = build_torch_encoder(encoder, enc_cfg, device, enc_cfg.get("checkpoint"))
    student.train()
    teacher = copy.deepcopy(student)
    teacher.eval()
    for p in teacher.model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(student.model.parameters(), lr=float(train_cfg.get("lr", 1e-5)),
                            weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    epochs = int(train_cfg.get("epochs", 1))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    momentum = float(dapt_cfg.get("ema_momentum", 0.999))
    mask_frac = float(dapt_cfg.get("mask_frac", 0.4))
    n_spans = int(dapt_cfg.get("mask_spans", 8))
    rng = torch.Generator().manual_seed(int(train_cfg.get("seed", 42)))

    step = 0
    history = []
    for epoch in range(epochs):
        running = 0.0
        for wav in tqdm(loader, desc=f"dapt/{encoder} ep{epoch}", leave=False):
            wav_masked = _time_mask(wav, mask_frac, n_spans, rng)
            z_s = pooled_features(student, encoder, wav_masked)
            with torch.no_grad():
                z_t = pooled_features(teacher, encoder, wav)
            loss = latent_bootstrap_loss(z_s, z_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.model.parameters(),
                                           max_norm=float(train_cfg.get("max_grad_norm", 1.0)))
            opt.step()
            sched.step()
            _ema_update(teacher, student, momentum)
            running += loss.item()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        avg = running / max(1, len(loader))
        history.append({"epoch": epoch, "loss": avg})
        logger.info("epoch %d: loss=%.4f", epoch, avg)
        if max_steps is not None and step >= max_steps:
            break

    state_path = output_dir / "state_dict.pt"
    torch.save(student.model.state_dict(), state_path)
    metrics = {
        "encoder": encoder, "variant": cfg.get("variant", "dapt"),
        "epochs": epochs, "steps": step, "ema_momentum": momentum,
        "mask_frac": mask_frac, "history": history, "state_dict": str(state_path),
    }
    with open(output_dir / "dapt_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved DAPT state_dict -> %s", state_path)
    return metrics
