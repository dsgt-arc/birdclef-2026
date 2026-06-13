"""SSAMBA native DAPT — masked-spectrogram SSL on the soundscape pool.

This replaces the generic pooled latent-bootstrap (`dapt.py`) with SSAMBA's
*native* objective: masked patch generation (MPG, per-patch normalized MSE) plus
masked patch classification (MPC, cross-sample NCE) — the loss SSAMBA was
pretrained with. The masking / MPG / MPC logic is copied from exp 103's
`SSAMBAPretrainModel.forward` (the venue recipe), but driven through the **exp 300
wrapper's Kaldi-fbank frontend at the wrapper's exact eval geometry** so the trained
backbone's `state_dict` loads straight back into `SSAMBAHFEncoder.model`.

Why not import exp 103's class: it is welded to nnAudio / 32 kHz / 224-mel / 224×1
patches (a different audio space). We continue-pretrain *from* `ssamba_small_400.pth`
at 16 kHz / 128-mel / 16×16 patches — the geometry exp 300 evaluates — so the only
reuse is the masking + head + loss *logic*, reimplemented over the wrapper frontend.

Unlike the degenerate latent bootstrap, this objective is non-degenerate by
construction: the target is the *raw masked mel patch* (not a mask-invariant pooled
vector), so a masked position cannot be solved by averaging visible patches.

Two trainable-parameter arms (config `peft`):
  - ``none`` (full DAPT) — train all backbone weights, save ``state_dict.pt``
    (exp 300 consumes via ``--state-dict``).
  - ``lora`` — freeze the base, inject LoRA into ``[in_proj, out_proj]`` (validated
    in the smoke), train adapters + heads, save the adapter dir (``--adapter``).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from birdclef_300.registry import build_torch_encoder
from birdclef_301.data import SoundscapeDataset

logger = logging.getLogger("birdclef_301.ssamba_dapt")


class _SSAMBAHeads(nn.Module):
    """MPG (MSE recon) + MPC (NCE) heads + learnable mask embedding.

    Logic copied from exp 103 ``SSAMBAPretrainModel.__init__`` (decoder_depth=2).
    These are discarded at save time — only the backbone is reloaded at eval.
    """

    def __init__(self, embed_dim: int, patch_dim: int, decoder_depth: int = 2):
        super().__init__()

        def _mk():
            if decoder_depth == 1:
                return nn.Linear(embed_dim, patch_dim)
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, patch_dim),
            )

        self.gpredlayer = _mk()  # MPG: generative (reconstruction)
        self.cpredlayer = _mk()  # MPC: discriminative (contrastive)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_normal_(self.mask_embed)


def _generate_span_mask(num_patches: int, mask_ratio: float, span_min: int,
                        span_max: int, device: torch.device) -> torch.Tensor:
    """Contiguous span mask over all `num_patches` (full 5 s window, no padding).

    Copied from exp 103 ``_generate_span_mask`` with num_real == num_total.
    """
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    target_masked = int(num_patches * mask_ratio)
    while mask.sum() < target_masked:
        span_len = random.randint(span_min, span_max)
        max_start = num_patches - span_len
        if max_start < 0:
            span_len, max_start = num_patches, 0
        start = random.randint(0, max_start)
        mask[start:start + span_len] = True
    return mask


def _generate_random_mask(num_patches: int, mask_ratio: float,
                          device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    mask[random.sample(range(num_patches), int(num_patches * mask_ratio))] = True
    return mask


def _normalize_patches(patches: torch.Tensor) -> torch.Tensor:
    """Per-patch normalization (exp 103 ``_normalize_patches``)."""
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (patches - mean) / std


def ssamba_masked_forward(backbone, heads: _SSAMBAHeads, unfold: nn.Unfold,
                          mel: torch.Tensor, dapt_cfg: dict,
                          mask_bool: torch.Tensor | None = None) -> dict:
    """One masked-SSL forward over a mel tensor `(B, 1, F, T)`.

    Mirrors ``SSAMBAModel.forward_features`` exactly (middle-CLS insertion + the
    documented double ``norm_f`` for eval parity), with masked patch substitution
    before the backbone — i.e. the masked re-implementation of the eval path, so the
    DAPT'd weights are optimized for the readout exp 300 actually uses.

    ``mask_bool`` (optional ``(N,)`` bool) overrides the random mask — used by the
    validation overfit test for a deterministic signal.
    """
    B = mel.shape[0]
    device = mel.device

    # Raw patch targets (freq-first order, matches Conv2d.flatten) — the native target.
    raw_patches = unfold(mel).transpose(1, 2)            # (B, N, patch_dim)
    patch_embeddings = backbone.patch_embed(mel)         # (B, N, D)
    num_patches = patch_embeddings.shape[1]

    if mask_bool is None:
        if dapt_cfg.get("mask_mode", "span") == "span":
            mask_bool = _generate_span_mask(
                num_patches, float(dapt_cfg.get("mask_ratio", 0.4)),
                int(dapt_cfg.get("mask_span_min", 5)), int(dapt_cfg.get("mask_span_max", 15)),
                device)
        else:
            mask_bool = _generate_random_mask(num_patches, float(dapt_cfg.get("mask_ratio", 0.4)), device)
    mask_idx = mask_bool.nonzero(as_tuple=True)[0]
    M = mask_idx.shape[0]

    # Substitute the learnable mask embedding at masked positions.
    masked = patch_embeddings.clone()
    mask_embed = heads.mask_embed.to(dtype=masked.dtype)
    masked[:, mask_idx, :] = mask_embed.expand(B, M, -1)

    # Patch pos-embed + middle-CLS insertion (exactly SSAMBAModel.forward_features).
    x = masked + backbone.pos_embed[:, 1:, :]
    cls_pos = None
    if backbone.cls_token is not None:
        cls = (backbone.cls_token + backbone.pos_embed[:, :1, :]).expand(B, -1, -1)
        cls_pos = num_patches // 2 if backbone.config.use_middle_cls_token else 0
        if cls_pos == 0:
            x = torch.cat([cls, x], dim=1)
        else:
            x = torch.cat([x[:, :cls_pos, :], cls, x[:, cls_pos:, :]], dim=1)
    x = backbone.pos_drop(x)

    hidden_states, residual = backbone.layers(x)
    residual = hidden_states if residual is None else residual + backbone.drop_path(hidden_states)
    x = backbone.norm_f(residual.to(dtype=backbone.norm_f.weight.dtype))
    x = backbone.norm_f(x)  # double norm_f — eval-path parity (see forward_features note)

    # Strip CLS to recover patch representations.
    if cls_pos is None:
        patch_repr = x
    elif cls_pos == 0:
        patch_repr = x[:, 1:, :]
    else:
        patch_repr = torch.cat([x[:, :cls_pos, :], x[:, cls_pos + 1:, :]], dim=1)
    masked_repr = patch_repr[:, mask_idx, :]             # (B, M, D)

    # MPG: per-patch normalized MSE reconstruction at masked positions.
    mpg_pred = heads.gpredlayer(masked_repr)
    mpg_targets = raw_patches[:, mask_idx, :]
    if dapt_cfg.get("normalize_targets", True):
        mpg_targets = _normalize_patches(mpg_targets)
    loss_mpg = F.mse_loss(mpg_pred, mpg_targets)

    # MPC: cross-sample NCE with a shared negative pool (token-level).
    raw_masked = raw_patches[:, mask_idx, :].detach()
    nce_pred = heads.cpredlayer(masked_repr)
    BN = B * M
    K = min(int(dapt_cfg.get("nce_num_negatives", 100)), BN - 1)
    temp = float(dapt_cfg.get("nce_temperature", 0.1))
    with torch.amp.autocast("cuda", enabled=False):
        pred_flat = F.normalize(nce_pred.float().reshape(BN, -1), dim=-1)
        tgt_flat = F.normalize(raw_masked.float().reshape(BN, -1), dim=-1)
        neg_pool = tgt_flat[torch.randperm(BN, device=device)[:K]]
        pos_sim = (pred_flat * tgt_flat).sum(dim=-1, keepdim=True)
        neg_sim = pred_flat @ neg_pool.t()
        logits = torch.cat([pos_sim, neg_sim], dim=-1) / temp
        labels = torch.zeros(BN, dtype=torch.long, device=device)
        loss_nce = F.cross_entropy(logits, labels)
    with torch.no_grad():
        nce_acc = (logits.argmax(dim=-1) == 0).float().mean()

    return {"loss_mpg": loss_mpg, "loss_nce": loss_nce,
            "nce_accuracy": nce_acc, "mask_count": M, "num_patches": num_patches}


def run_ssamba_dapt(cfg: dict, output_dir: str | Path, max_steps: int | None = None) -> dict:
    enc_cfg = cfg["encoder_cfg"]
    dapt_cfg = cfg.get("dapt", {})
    train_cfg = cfg.get("training", {})
    peft_mode = cfg.get("peft", "none")
    device = enc_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; using CPU")
        device = "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = build_torch_encoder("ssamba_hf", enc_cfg, device, enc_cfg.get("checkpoint"))
    backbone = wrapper.model           # the SSAMBAModel whose state_dict exp 300 reloads
    backbone.train()
    cfgm = backbone.config
    patch_dim = cfgm.fpatch * cfgm.tpatch
    heads = _SSAMBAHeads(cfgm.embed_dim, patch_dim,
                         int(dapt_cfg.get("decoder_depth", 2))).to(device)
    unfold = nn.Unfold(kernel_size=(cfgm.fpatch, cfgm.tpatch),
                       stride=(cfgm.fpatch, cfgm.tpatch)).to(device)

    peft_model = None
    if peft_mode == "lora":
        from peft import LoraConfig, get_peft_model
        lora_cfg = cfg["lora"]
        peft_model = get_peft_model(backbone, LoraConfig(
            r=int(lora_cfg["rank"]),
            lora_alpha=int(lora_cfg.get("alpha", 2 * int(lora_cfg["rank"]))),
            lora_dropout=float(lora_cfg.get("dropout", 0.1)),
            target_modules=list(lora_cfg.get("target_modules", ["in_proj", "out_proj"])),
            bias="none",
        ))  # injects LoRA into backbone submodules in place
        peft_model.print_trainable_parameters()
        params = [p for p in backbone.parameters() if p.requires_grad] + list(heads.parameters())
    else:
        params = list(backbone.parameters()) + list(heads.parameters())

    opt = torch.optim.AdamW(params, lr=float(train_cfg.get("lr", 1e-5)),
                            weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    ds = SoundscapeDataset(cfg["data"]["soundscape_dir"])
    loader = DataLoader(ds, batch_size=int(train_cfg.get("batch_size", 16)), shuffle=True,
                        drop_last=True, num_workers=int(train_cfg.get("num_workers", 4)))
    logger.info("SSAMBA native DAPT (peft=%s): %d soundscape files", peft_mode, len(ds))
    epochs = int(train_cfg.get("epochs", 1))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    mse_w = float(dapt_cfg.get("mse_weight", 10.0))
    nce_w = float(dapt_cfg.get("nce_weight", 1.0))
    random.seed(int(train_cfg.get("seed", 42)))

    step = 0
    history = []
    for epoch in range(epochs):
        run_mpg = run_nce = run_acc = 0.0
        for wav in tqdm(loader, desc=f"ssamba-dapt ep{epoch}", leave=False):
            x = wav.squeeze(1).to(device)
            fbank = wrapper._waveform_to_fbank(x)              # (B, T, F) — exact eval frontend
            mel = fbank.unsqueeze(1).transpose(2, 3)           # (B, 1, F, T) — matches forward()
            out = ssamba_masked_forward(backbone, heads, unfold, mel, dapt_cfg)
            loss = mse_w * out["loss_mpg"] + nce_w * out["loss_nce"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(train_cfg.get("max_grad_norm", 1.0)))
            opt.step()
            sched.step()
            run_mpg += out["loss_mpg"].item()
            run_nce += out["loss_nce"].item()
            run_acc += out["nce_accuracy"].item()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        n = max(1, min(len(loader), step))
        history.append({"epoch": epoch, "loss_mpg": run_mpg / n,
                        "loss_nce": run_nce / n, "nce_accuracy": run_acc / n})
        logger.info("epoch %d: mpg=%.4f nce=%.4f nce_acc=%.3f",
                    epoch, run_mpg / n, run_nce / n, run_acc / n)
        if max_steps is not None and step >= max_steps:
            break

    variant = cfg.get("variant", "dapt-native-ep1")
    if peft_mode == "lora":
        out_path = output_dir / "adapter"
        peft_model.save_pretrained(str(out_path))
    else:
        out_path = output_dir / "state_dict.pt"
        torch.save(backbone.state_dict(), out_path)
    metrics = {"encoder": "ssamba_hf", "variant": variant, "peft": peft_mode,
               "objective": "mpg+mpc", "epochs": epochs, "steps": step,
               "mse_weight": mse_w, "nce_weight": nce_w, "history": history,
               "output": str(out_path)}
    with open(output_dir / "dapt_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved SSAMBA native DAPT (%s) -> %s", peft_mode, out_path)
    return metrics
