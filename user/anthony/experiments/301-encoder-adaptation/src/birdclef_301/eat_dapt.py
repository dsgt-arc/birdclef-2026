"""EAT native DAPT — from-scratch PyTorch data2vec-2.0 port (no fairseq).

EAT's native objective is data2vec-2.0 self-distillation: an EMA *teacher* sees the
full spectrogram and produces per-patch targets (the average of every transformer
block's FFN output, ``t`` from ``AltBlock``); a *student* sees only the visible
(unmasked) patches, and a lightweight decoder reconstructs the teacher's targets at
the masked positions (frame-level ``d2v_loss``, MSE). This is the venue recipe
(Baevski et al., data2vec 2.0, ICML 2023; EAT, Chen et al. 2024) — not the degenerate
pooled latent-bootstrap.

We port it directly over the HF model's PyTorch ViT (``wrapper.model.model`` is an
``EAT`` nn.Module: ``local_encoder`` patch conv, ``extra_tokens`` CLS, frozen sincos
``fixed_positional_encoder``, ``pre_norm``, 12× ``AltBlock``). We write our own
student/teacher forward; the HF ``encode``/``extract_features`` are used only by
exp 300 at eval. The saved ``state_dict`` is the ``EAT`` module's, so it reloads into
``EATModel...model.model`` via exp 300's ``--state-dict`` overlay.

Smallest faithful subset shipped here: frame-level ``d2v_loss`` only. Deferred:
``cls_loss`` (utterance term), the exact conv ``Decoder2d``, ``clone_batch``
multi-mask, alibi, and block (vs random) masking — all logged as dropped.

Two arms (config ``peft``): ``none`` (full, save ``state_dict.pt``) and ``lora``
(freeze base, adapters on ``[qkv, proj]``, save adapter dir). The EMA teacher is a
deepcopy of the post-PEFT student, so for the LoRA arm the teacher EMA-tracks the
adapter deltas with the base frozen.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from birdclef_300.registry import build_torch_encoder
from birdclef_301.data import SoundscapeDataset

logger = logging.getLogger("birdclef_301.eat_dapt")


# --------------------------------------------------------------------------- #
# Forward building blocks (imported by the validation tests)
# --------------------------------------------------------------------------- #
def eat_teacher_targets(eat, mel: torch.Tensor, top_k: int = 12,
                        layer_norm_targets: bool = True) -> torch.Tensor:
    """Full (unmasked) forward; target = layer_norm(mean of top-k blocks' FFN ``t``).

    Replicates ``EAT.encode`` but captures each ``AltBlock``'s second return value
    ``t`` (the FFN target — for ``layer_norm_first=False`` this is
    ``mlp(norm1(x + attn(x)))``). CLS is dropped → per-patch targets ``(B, P, D)``.
    Caller is responsible for ``torch.no_grad()`` / teacher eval mode.
    """
    B = mel.shape[0]
    p = eat.local_encoder(mel)
    if eat.fixed_positional_encoder is not None:
        p = p + eat.fixed_positional_encoder(p, None)[:, :p.size(1), :]
    x = torch.cat((eat.extra_tokens.expand(B, -1, -1), p), dim=1)
    x = eat.pre_norm(x)
    x = eat.pos_drop(x)
    targets = []
    for blk in eat.blocks:
        x, t = blk(x)
        targets.append(t[:, 1:])                      # drop CLS → per-patch
    y = torch.stack(targets[-top_k:], dim=0).mean(dim=0)   # (B, P, D)
    if layer_norm_targets:
        y = F.layer_norm(y, (y.shape[-1],))
    return y


def eat_student_encode_visible(eat, mel: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    """Encode only the visible patches (data2vec-2.0 efficient encoder).

    ``keep_idx``: ``(B, n_keep)`` patch indices to keep. Pos-embed is added to all
    patches *before* gathering (so each visible patch keeps its absolute position),
    then CLS is prepended and the blocks run on the short sequence.
    Returns ``(B, 1 + n_keep, D)``.
    """
    B = mel.shape[0]
    p = eat.local_encoder(mel)
    if eat.fixed_positional_encoder is not None:
        p = p + eat.fixed_positional_encoder(p, None)[:, :p.size(1), :]
    D = p.size(-1)
    p_vis = torch.gather(p, 1, keep_idx.unsqueeze(-1).expand(-1, -1, D))
    x = torch.cat((eat.extra_tokens.expand(B, -1, -1), p_vis), dim=1)
    x = eat.pre_norm(x)
    x = eat.pos_drop(x)
    for blk in eat.blocks:
        x, _ = blk(x)
    return x


def make_random_masks(B: int, P: int, mask_ratio: float, device: torch.device,
                      generator: torch.Generator | None = None):
    """Per-sample random patch masking. Returns ``(keep_idx, mask_idx)``."""
    n_keep = max(1, int(round(P * (1.0 - mask_ratio))))
    perm = torch.argsort(torch.rand(B, P, device=device, generator=generator), dim=1)
    return perm[:, :n_keep].contiguous(), perm[:, n_keep:].contiguous()


class D2vDecoder(nn.Module):
    """Lightweight conv decoder (data2vec-2.0 "lightweight decoder" intent).

    Student visible-patch outputs are scattered back to their grid positions; masked
    positions get a learnable mask token; a small Conv1d stack over the patch
    sequence lets masked positions read nearby visible context, then a linear head
    predicts the teacher target dim. Discarded at save time (not in the eval wrapper).
    """

    def __init__(self, dim: int, num_patches: int, kernel: int = 5):
        super().__init__()
        self.num_patches = num_patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        pad = kernel // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.conv2 = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, vis_patches: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
        """``vis_patches``: ``(B, n_keep, D)`` student patch outputs (CLS dropped)."""
        B, _, D = vis_patches.shape
        full = self.mask_token.expand(B, self.num_patches, D).clone()
        full.scatter_(1, keep_idx.unsqueeze(-1).expand(-1, -1, D), vis_patches)
        full = full + self.pos
        h = full.transpose(1, 2)                      # (B, D, P)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = h.transpose(1, 2)                         # (B, P, D)
        return self.head(self.norm(h))


def eat_d2v_loss(eat_student, eat_teacher, decoder: D2vDecoder, mel: torch.Tensor,
                 keep_idx: torch.Tensor, mask_idx: torch.Tensor, top_k: int = 12,
                 layer_norm_targets: bool = True) -> dict:
    """One data2vec frame-level forward → masked-position MSE + diagnostics."""
    with torch.no_grad():
        y = eat_teacher_targets(eat_teacher, mel, top_k, layer_norm_targets)   # (B, P, D)
    sx = eat_student_encode_visible(eat_student, mel, keep_idx)                # (B, 1+n_keep, D)
    pred_full = decoder(sx[:, 1:, :], keep_idx)                                # (B, P, D)
    D = y.size(-1)
    pred_m = torch.gather(pred_full, 1, mask_idx.unsqueeze(-1).expand(-1, -1, D))
    tgt_m = torch.gather(y, 1, mask_idx.unsqueeze(-1).expand(-1, -1, D))
    loss = (F.mse_loss(pred_m, tgt_m, reduction="none") / math.sqrt(D)).mean()
    return {"loss": loss, "target_var": y.var().detach(),
            "pred_var": pred_full.var().detach(),
            "target_mean_mse": F.mse_loss(tgt_m.mean(dim=1, keepdim=True).expand_as(tgt_m), tgt_m).detach(),
            "zero_mse": tgt_m.pow(2).mean().detach()}


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(decay).add_(sp.detach(), alpha=1.0 - decay)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


def build_eat_dapt(cfg: dict, device: str):
    """Build (wrapper, eat_student, eat_teacher, decoder, params, peft_model).

    Factored out so the validation tests build the exact same objects.
    """
    enc_cfg = cfg["encoder_cfg"]
    peft_mode = cfg.get("peft", "none")
    wrapper = build_torch_encoder("eat", enc_cfg, device, enc_cfg.get("model_name"))
    eat = wrapper.model.model            # the EAT nn.Module whose state_dict exp 300 reloads
    eat.train()

    peft_model = None
    if peft_mode == "lora":
        from peft import LoraConfig, get_peft_model
        lora_cfg = cfg["lora"]
        peft_model = get_peft_model(eat, LoraConfig(
            r=int(lora_cfg["rank"]),
            lora_alpha=int(lora_cfg.get("alpha", 2 * int(lora_cfg["rank"]))),
            lora_dropout=float(lora_cfg.get("dropout", 0.1)),
            target_modules=list(lora_cfg.get("target_modules", ["qkv", "proj"])),
            bias="none",
        ))
        peft_model.print_trainable_parameters()

    # P = (target_length / patch) * (mel_bins / patch)
    c = eat.config
    P = (c.img_size[0] // c.patch_size) * (c.mel_bins // c.patch_size) \
        if isinstance(c.img_size, (tuple, list)) else (c.max_length // c.patch_size) * (c.mel_bins // c.patch_size)
    # Robust fallback: derive P from a dummy forward at the wrapper's target_length.
    target_len = int(enc_cfg.get("target_length", 512))
    P = (target_len // c.patch_size) * (c.mel_bins // c.patch_size)

    decoder = D2vDecoder(c.embed_dim, P).to(device)
    teacher = copy.deepcopy(eat)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable = [p for p in eat.parameters() if p.requires_grad]
    params = trainable + list(decoder.parameters())
    return wrapper, eat, teacher, decoder, params, peft_model, P


def run_eat_dapt(cfg: dict, output_dir: str | Path, max_steps: int | None = None) -> dict:
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

    wrapper, eat, teacher, decoder, params, peft_model, P = build_eat_dapt(cfg, device)

    opt = torch.optim.AdamW(params, lr=float(train_cfg.get("lr", 1e-5)),
                            weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    # `window_samples` (optional) overrides the 5 s default — e.g. 160000 feeds EAT 10 s
    # of real audio. Paired with encoder_cfg.target_length=1024 this matches EAT's native
    # 10 s pretraining geometry (vs the 5 s/512 default); the wrapper's _waveform_to_mel
    # crops/pads to target_length and P is derived from it above, so this is a pure config flip.
    ds_kwargs = {}
    if "window_samples" in cfg["data"]:
        ds_kwargs["window_samples"] = int(cfg["data"]["window_samples"])
    # `val_frac` > 0 holds out that fraction of soundscape FILES to score the same d2v
    # objective each epoch (held-out SSL loss) — the train/val gap is the overfitting signal.
    val_frac = float(cfg["data"].get("val_frac", 0.0))
    bs = int(train_cfg.get("batch_size", 12))
    nw = int(train_cfg.get("num_workers", 4))
    ds = SoundscapeDataset(cfg["data"]["soundscape_dir"],
                           split=("train" if val_frac > 0 else "all"), val_frac=val_frac, **ds_kwargs)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw)
    val_loader = None
    if val_frac > 0:
        val_ds = SoundscapeDataset(cfg["data"]["soundscape_dir"], split="val", val_frac=val_frac, **ds_kwargs)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw)
        logger.info("EAT native DAPT (peft=%s): %d train / %d val files, %d patches",
                    peft_mode, len(ds), len(val_ds), P)
    else:
        logger.info("EAT native DAPT (peft=%s): %d soundscape files, %d patches", peft_mode, len(ds), P)
    epochs = int(train_cfg.get("epochs", 1))
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    mask_ratio = float(dapt_cfg.get("mask_ratio", 0.8))
    top_k = int(dapt_cfg.get("top_k", 12))
    ln_tgt = bool(dapt_cfg.get("layer_norm_targets", True))
    m0 = float(dapt_cfg.get("ema_decay", 0.999))
    m1 = float(dapt_cfg.get("ema_end_decay", 0.9999))
    gen = torch.Generator(device=device).manual_seed(int(train_cfg.get("seed", 42)))

    def _val_pass():
        """Score the identical d2v objective on held-out files (no grad). A fresh
        fixed-seed mask generator keeps val comparable epoch-to-epoch and isolated
        from the train mask stream (`gen`)."""
        if val_loader is None:
            return {}
        vgen = torch.Generator(device=device).manual_seed(int(train_cfg.get("seed", 42)))
        eat.eval()
        vl = vt = vp = 0.0; nb = 0
        with torch.no_grad():
            for wav in val_loader:
                mel = wrapper._waveform_to_mel(wav.squeeze(1)).to(device)
                keep_idx, mask_idx = make_random_masks(mel.shape[0], P, mask_ratio, device, vgen)
                out = eat_d2v_loss(eat, teacher, decoder, mel, keep_idx, mask_idx, top_k, ln_tgt)
                vl += out["loss"].item(); vt += out["target_var"].item(); vp += out["pred_var"].item(); nb += 1
        eat.train()
        nb = max(1, nb)
        return {"val_loss": vl / nb, "val_target_var": vt / nb, "val_pred_var": vp / nb}

    step = 0
    history = []
    for epoch in range(epochs):
        run_loss = run_tvar = run_pvar = 0.0
        for wav in tqdm(loader, desc=f"eat-dapt ep{epoch}", leave=False):
            mel = wrapper._waveform_to_mel(wav.squeeze(1)).to(device)   # (B, 1, 512, 128)
            B = mel.shape[0]
            keep_idx, mask_idx = make_random_masks(B, P, mask_ratio, device, gen)
            out = eat_d2v_loss(eat, teacher, decoder, mel, keep_idx, mask_idx, top_k, ln_tgt)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(train_cfg.get("max_grad_norm", 1.0)))
            opt.step()
            sched.step()
            decay = m0 + (m1 - m0) * min(1.0, step / max(1, total))
            ema_update(teacher, eat, decay)
            run_loss += loss.item()
            run_tvar += out["target_var"].item()
            run_pvar += out["pred_var"].item()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        n = max(1, min(len(loader), step))
        val = _val_pass()
        history.append({"epoch": epoch, "loss": run_loss / n,
                        "target_var": run_tvar / n, "pred_var": run_pvar / n, **val})
        logger.info("epoch %d: d2v_loss=%.5f target_var=%.3f pred_var=%.3f | val_loss=%.5f val_pred_var=%.3f",
                    epoch, run_loss / n, run_tvar / n, run_pvar / n,
                    val.get("val_loss", float("nan")), val.get("val_pred_var", float("nan")))
        if max_steps is not None and step >= max_steps:
            break

    variant = cfg.get("variant", "dapt-native-ep1")
    if peft_mode == "lora":
        out_path = output_dir / "adapter"
        peft_model.save_pretrained(str(out_path))
    else:
        out_path = output_dir / "state_dict.pt"
        # Save the OUTER EATModel state_dict (model.-prefixed keys): exp 300's
        # _apply_overlay loads via `wrapper.model.load_state_dict(...)`, and
        # wrapper.model is the EATModel wrapping our trained `eat` in place.
        torch.save(wrapper.model.state_dict(), out_path)
    metrics = {"encoder": "eat", "variant": variant, "peft": peft_mode,
               "objective": "data2vec-d2v", "epochs": epochs, "steps": step,
               "mask_ratio": mask_ratio, "top_k": top_k, "history": history,
               "output": str(out_path),
               "deferred": ["cls_loss", "Decoder2d", "clone_batch", "alibi", "block_masking"]}
    with open(output_dir / "dapt_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved EAT native DAPT (%s) -> %s", peft_mode, out_path)
    return metrics
