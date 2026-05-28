---
experiment: "034-pretrained-backbone"
status: in-progress
date: 2026-05-28
author: Murilo Gustineli
---

# Results — Exp 034 Pretrained-backbone (in-domain XC pretrain reuse)

Reuse the BirdCLEF+ 2025 2nd-place published in-domain-pretrained backbone
(EffNetV2-S / NFNet-L0, pretrained on 819K XC recordings) + our own SED head,
fine-tuned on BirdCLEF 2026. Baseline to beat: `distilled-sed-kd` fold 0 = **LB 0.906**.

Naming note: the code prefix is `xcpretrain` (names the lever — Xeno-Canto
in-domain pretraining). Earlier scaffold used the codename "yoink"; renamed
2026-05-28. Attribution to the source author stays in `proposal.md` § Provenance.

## Phase status

| Phase | What | Status |
|---|---|---|
| A | Source read + verified config (`docs/sydorskyi-config-verified.md`) | ✅ DONE (5 corrections) |
| B.1–B.5 | Forward path: mel + SED head + backbone loader + orchestrator (+35 tests) | ✅ DONE |
| B.6–B.9 | Training path: focal loss + sampler + augmentation + dataset + Lightning | ⬜ PENDING |
| **C — Gate 1** | **Feature parity vs vendored source** | ✅ **PASS (bit-identical)** |
| D — Gate 2 | Frozen-backbone head sanity (OOF AUC ≥0.95) | ⬜ next |
| E → J | Frozen FT + LB → 5-fold → full-FT → PP/TTA → ensemble → final | ⬜ |

## Phase C — Gate 1: feature parity (2026-05-28) ✅ PASS

**Method**: ran our `XCPretrainWaveClassifier` against Sydorskyi's own vendored
`WaveCNNAttenClasifier` (`vendor/sydorskyi-2025/` @ `053594f`) on 5 fixed
`train_soundscapes` audios, locally. Both fed the same waveform tensor; both
backbones loaded from the same cleaned pretrained `state_dict`. Compared at the
mel stage (`return_spec_feature`) and the backbone-embedding stage
(`return_cnn_emb`). Script: `scripts/gate1_parity.py`.

| Backbone | Out ch | Mel cosine / max-abs-diff | Emb cosine / max-abs-diff | Verdict |
|---|---|---|---|---|
| `tf_efficientnetv2_s_in21k` (2024 ckpt) | 256 | 1.00000000 / 0.0 | 1.00000000 / 0.0 | **PASS** |
| `eca_nfnet_l0` (2025 ckpt) | 2304 | 1.00000000 / 0.0 | 1.00000000 / 0.0 | **PASS** |

Result: **bit-identical** on all 5 audios, both stages, both backbones — far
beyond the cosine ≥0.999 / L2 <1e-4 gate. Confirms our B.1–B.5 port faithfully
reproduces Sydorskyi's mel + backbone forward path. Clears the research's
highest-probability catastrophic-bug class (nnAudio mel mismatch −0.05/−0.15;
standardize→min-max order/axes; head dim; ckpt keys). **No LB slot burned.**

Caveat (by design): this tests PORT fidelity, not library version-skew vs
Sydorskyi's original training environment. Version-skew is a lower-risk concern
that fine-tuning absorbs; Gate 2 (frozen-backbone OOF AUC) is the test that the
loaded backbone produces useful features for 2026.

## LB results

_(none yet — Phase C/D are offline gates; first LB is Phase E.)_
