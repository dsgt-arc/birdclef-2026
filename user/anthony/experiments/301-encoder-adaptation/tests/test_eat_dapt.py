"""Validation gate for the EAT data2vec port (`birdclef_301.eat_dapt`).

These run on a GPU node via `sbatch sbatch/validate-eat.sbatch` (they instantiate the
real HF EAT model). Two groups:

  Numerical stability — the port computes what we think it computes and does not
  collapse: FFN-target semantics, target shape/finiteness, anti-collapse variance,
  non-degeneracy at init, finite grads, state_dict round-trip.

  Directional learning — it actually learns the right thing: a tiny fixed-batch
  overfit drops the loss, beats trivial baselines (target-mean / zeros), and moves
  the representation measurably but boundedly.

These are the go/no-go before any full EAT DAPT run. They are deliberately strict:
the port is from-scratch, so a silently-wrong teacher target or a dead gradient must
fail loudly here, not surface as a null Δ after hours of training.
"""

from __future__ import annotations

import copy

import pytest
import torch

eat_dapt = pytest.importorskip("birdclef_301.eat_dapt")
from birdclef_301.eat_dapt import (  # noqa: E402
    D2vDecoder,
    build_eat_dapt,
    eat_d2v_loss,
    eat_teacher_targets,
    make_random_masks,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 768
TARGET_LEN = 512
N_PATCHES = (512 // 16) * (128 // 16)  # 32 * 8 = 256


def _cfg(peft: str = "none") -> dict:
    cfg = {
        "encoder_cfg": {"model_name": "worstchan/EAT-base_epoch30_pretrain",
                        "target_length": TARGET_LEN, "device": DEVICE},
        "peft": peft,
        "dapt": {"mask_ratio": 0.8, "top_k": 12, "layer_norm_targets": True},
    }
    if peft == "lora":
        cfg["lora"] = {"rank": 8, "target_modules": ["qkv", "proj"]}
    return cfg


def _wav(B: int = 2) -> torch.Tensor:
    """Random 5 s @ 16 kHz waveforms `(B, samples)` (deterministic)."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, 16000 * 5, generator=g)


@pytest.fixture(scope="module")
def built():
    """Build the EAT DAPT objects once (model load is expensive)."""
    wrapper, eat, teacher, decoder, params, peft_model, P = build_eat_dapt(_cfg(), DEVICE)
    return {"wrapper": wrapper, "eat": eat, "teacher": teacher,
            "decoder": decoder, "params": params, "P": P}


# ---------------------------- numerical stability --------------------------- #
def test_ffn_target_semantics(built):
    """[most load-bearing] AltBlock's returned `t` is the FFN target we average.

    For layer_norm_first=False: t == mlp(norm1(x + attn(x))). The whole teacher
    target rests on this — verify by hand-recompute on a real block.
    """
    eat = built["eat"]
    blk = eat.blocks[0].eval()
    assert eat.config.layer_norm_first is False, "port assumes layer_norm_first=False"
    x = torch.randn(2, N_PATCHES + 1, EMBED_DIM, device=DEVICE)
    with torch.no_grad():
        x_out, t = blk(x)
        a = x + blk.attn(x)              # drop_path is Identity (drop_path_rate=0)
        t_hand = blk.mlp(blk.norm1(a))
    assert torch.allclose(t, t_hand, atol=1e-5, rtol=1e-4), \
        f"FFN target mismatch: max|Δ|={ (t - t_hand).abs().max().item():.2e}"


def test_teacher_target_shape_and_finite(built):
    eat, wrapper = built["eat"], built["wrapper"]
    mel = wrapper._waveform_to_mel(_wav(2)).to(DEVICE)
    with torch.no_grad():
        y = eat_teacher_targets(eat.eval(), mel, top_k=12, layer_norm_targets=True)
    assert y.shape == (2, N_PATCHES, EMBED_DIM), y.shape
    assert torch.isfinite(y).all()


def test_anti_collapse_variance(built):
    """data2vec min_target_var / min_pred_var guards — fail loud on collapse."""
    eat, decoder, wrapper, P = built["eat"], built["decoder"], built["wrapper"], built["P"]
    mel = wrapper._waveform_to_mel(_wav(2)).to(DEVICE)
    keep, mask = make_random_masks(2, P, 0.8, DEVICE)
    out = eat_d2v_loss(eat, copy.deepcopy(eat).eval(), decoder, mel, keep, mask)
    assert out["target_var"].item() > 0.1, out["target_var"].item()
    assert out["pred_var"].item() > 0.01, out["pred_var"].item()


def test_nondegenerate_at_init(built):
    """Student≡teacher → masked loss is meaningfully > 0 (vs pooled bootstrap ~7e-4).

    Proves masked positions are genuinely hidden (the property the degenerate pooled
    objective lacked — there a mask-invariant pooled target gave ~0 loss at init).
    """
    eat, decoder, wrapper, P = built["eat"], built["decoder"], built["wrapper"], built["P"]
    teacher = copy.deepcopy(eat).eval()
    mel = wrapper._waveform_to_mel(_wav(2)).to(DEVICE)
    keep, mask = make_random_masks(2, P, 0.8, DEVICE)
    out = eat_d2v_loss(eat, teacher, decoder, mel, keep, mask)
    assert out["loss"].item() > 1e-2, f"loss at init too small ({out['loss'].item():.2e}) — degenerate?"


def test_finite_grads_reach_blocks(built):
    eat, decoder, wrapper, P = built["eat"], built["decoder"], built["wrapper"], built["P"]
    teacher = copy.deepcopy(eat).eval()
    mel = wrapper._waveform_to_mel(_wav(2)).to(DEVICE)
    keep, mask = make_random_masks(2, P, 0.8, DEVICE)
    eat.zero_grad(); decoder.zero_grad()
    eat_d2v_loss(eat, teacher, decoder, mel, keep, mask)["loss"].backward()
    gnorm = sum(p.grad.norm().item() for p in eat.blocks.parameters() if p.grad is not None)
    assert gnorm > 0.0 and torch.isfinite(torch.tensor(gnorm)), gnorm


def test_state_dict_round_trip(built):
    """The saved state_dict reloads via EXACTLY exp 300 `_apply_overlay`'s call.

    `_apply_overlay` does `wrapper.model.load_state_dict(torch.load(state_dict))`
    with the DEFAULT strict=True. For EAT, `wrapper.model` is the EATModel (keys are
    `model.`-prefixed), so we must save `wrapper.model.state_dict()` — NOT the inner
    `eat` module's. Mirror that load target and strictness here (an earlier version
    loaded into `.model.model` and silently passed while re-extract failed).
    """
    from birdclef_300.registry import build_torch_encoder
    sd = built["wrapper"].model.state_dict()      # what run_eat_dapt saves
    fresh = build_torch_encoder("eat", _cfg()["encoder_cfg"], DEVICE,
                                "worstchan/EAT-base_epoch30_pretrain")
    fresh.model.load_state_dict(sd)               # strict=True, mirrors _apply_overlay


# ---------------------------- directional learning -------------------------- #
@pytest.fixture(scope="module")
def overfit(built):
    """Overfit one fixed batch ~40 steps; return loss trajectory + baselines."""
    wrapper, eat, decoder, P = built["wrapper"], built["eat"], built["decoder"], built["P"]
    teacher = copy.deepcopy(eat).eval()
    mel = wrapper._waveform_to_mel(_wav(2)).to(DEVICE)
    keep, mask = make_random_masks(2, P, 0.8, DEVICE)  # fixed mask for a clean overfit signal
    params = list(eat.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    losses = []
    last = None
    for _ in range(40):
        out = eat_d2v_loss(eat, teacher, decoder, mel, keep, mask)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        losses.append(out["loss"].item())
        last = out
    return {"losses": losses, "last": last}


def test_tiny_overfit_loss_drops(overfit):
    losses = overfit["losses"]
    assert losses[-1] < 0.5 * losses[0], \
        f"overfit failed: start={losses[0]:.4f} end={losses[-1]:.4f}"


def test_beats_trivial_baselines(overfit):
    """Post-overfit masked MSE beats predicting-the-mean and predicting-zeros."""
    out = overfit["last"]
    assert out["loss"].item() < out["target_mean_mse"].item(), "no better than target-mean"
    assert out["loss"].item() < out["zero_mse"].item(), "no better than zeros"


def test_representation_moves_but_bounded(built, overfit):
    """Adapted pooled embedding shifts from the frozen base (cosine<1) yet stays
    finite/bounded. `built['eat']` was trained in place by the overfit fixture, so we
    compare its pooled output to a freshly-reloaded (frozen) EAT base."""
    wrapper, eat = built["wrapper"], built["eat"]
    mel = wrapper._waveform_to_mel(_wav(2)).to(DEVICE)
    base = build_torch_encoder_eat()  # frozen base, reloaded from HF
    with torch.no_grad():
        adapted = eat.eval().extract_features(mel)[:, 1:].mean(dim=1)   # (B, D)
        frozen = base.extract_features(_wav(2).unsqueeze(1).to(DEVICE))  # (B, D), mean_patch
    assert torch.isfinite(adapted).all()
    cos = torch.nn.functional.cosine_similarity(adapted, frozen, dim=-1).mean().item()
    assert cos < 0.9999, f"representation did not move (cos={cos:.6f})"
    assert adapted.norm(dim=-1).max().item() < 1e4, "representation exploded"


# ------------------------------- LoRA arm ----------------------------------- #
def test_lora_adapter_merges(tmp_path):
    """LoRA arm: adapters inject on [qkv, proj], train one step, merge_and_unload OK."""
    wrapper, eat, teacher, decoder, params, peft_model, P = build_eat_dapt(_cfg("lora"), DEVICE)
    assert peft_model is not None
    trainable = sum(p.numel() for p in eat.parameters() if p.requires_grad)
    assert trainable > 0, "no trainable LoRA params"
    mel = wrapper._waveform_to_mel(_wav(2)).to(DEVICE)
    keep, mask = make_random_masks(2, P, 0.8, DEVICE)
    opt = torch.optim.AdamW(params, lr=1e-4)
    out = eat_d2v_loss(eat, teacher, decoder, mel, keep, mask)
    opt.zero_grad(); out["loss"].backward(); opt.step()
    peft_model.save_pretrained(str(tmp_path / "adapter"))
    from peft import PeftModel
    base = build_torch_encoder_eat()
    merged = PeftModel.from_pretrained(base.model.model, str(tmp_path / "adapter")).merge_and_unload()
    assert merged is not None


def build_torch_encoder_eat():
    from birdclef_300.registry import build_torch_encoder
    return build_torch_encoder("eat", {"model_name": "worstchan/EAT-base_epoch30_pretrain",
                                       "target_length": TARGET_LEN, "device": DEVICE},
                               DEVICE, "worstchan/EAT-base_epoch30_pretrain")
