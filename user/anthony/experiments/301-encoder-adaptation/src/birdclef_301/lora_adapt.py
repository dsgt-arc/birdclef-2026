"""Supervised-focal LoRA adaptation (BEATs / EAT / SSAMBA-HF).

Builds the same shared encoder wrapper exp 300 uses, injects LoRA into its inner
`.model`, trains a focal multi-label head end-to-end, then saves the PEFT adapter
dir. Exp 300 re-extracts with `birdclef-300 extract --variant lora-rN --adapter <dir>`
and the ridge probe reports the frozen→adapted Δ (the head is discarded — the probe
replaces it).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from birdclef_300.registry import build_torch_encoder
from birdclef_301.data import FocalDataset
from birdclef_301.encoder_forward import pooled_features
from birdclef_301.losses import BCEFocalLoss

logger = logging.getLogger("birdclef_301.lora")


def _collate(batch):
    wavs, labels = zip(*batch)
    return torch.stack(wavs), torch.stack(labels)


def run_lora(cfg: dict, output_dir: str | Path, max_steps: int | None = None) -> dict:
    encoder = cfg["encoder"]
    enc_cfg = cfg["encoder_cfg"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg.get("training", {})
    device = enc_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; using CPU")
        device = "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = FocalDataset(cfg["data"]["train_csv"], cfg["data"]["audio_dir"])
    loader = DataLoader(
        ds, batch_size=int(train_cfg.get("batch_size", 16)), shuffle=True,
        drop_last=True, num_workers=int(train_cfg.get("num_workers", 4)), collate_fn=_collate,
    )
    logger.info("Focal dataset: %d recordings, %d species", len(ds), ds.num_classes)

    wrapper = build_torch_encoder(encoder, enc_cfg, device, enc_cfg.get("checkpoint"))
    wrapper.train()
    lora_config = LoraConfig(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg.get("alpha", 2 * int(lora_cfg["rank"]))),
        lora_dropout=float(lora_cfg.get("dropout", 0.1)),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    peft_model = get_peft_model(wrapper.model, lora_config)  # injects LoRA in place
    peft_model.print_trainable_parameters()

    head = nn.Linear(wrapper.embed_dim, ds.num_classes).to(device)
    params = [p for p in wrapper.model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=float(train_cfg.get("lr", 1e-4)), weight_decay=0.01)
    epochs = int(train_cfg.get("epochs", 5))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    crit = BCEFocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)))

    step = 0
    history = []
    for epoch in range(epochs):
        running = 0.0
        for wav, y in tqdm(loader, desc=f"lora/{encoder} ep{epoch}", leave=False):
            y = y.to(device)
            logits = head(pooled_features(wrapper, encoder, wav))
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(train_cfg.get("max_grad_norm", 1.0)))
            opt.step()
            sched.step()
            running += loss.item()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        avg = running / max(1, len(loader))
        history.append({"epoch": epoch, "loss": avg})
        logger.info("epoch %d: loss=%.4f", epoch, avg)
        if max_steps is not None and step >= max_steps:
            break

    adapter_dir = output_dir / "adapter"
    peft_model.save_pretrained(str(adapter_dir))
    torch.save(head.state_dict(), output_dir / "head.pt")
    metrics = {
        "encoder": encoder, "variant": cfg.get("variant", f"lora-r{lora_cfg['rank']}"),
        "rank": int(lora_cfg["rank"]), "epochs": epochs, "steps": step,
        "num_classes": ds.num_classes, "history": history, "adapter_dir": str(adapter_dir),
    }
    with open(output_dir / "lora_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved LoRA adapter -> %s", adapter_dir)
    return metrics
