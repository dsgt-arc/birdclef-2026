# Experiment 301: Encoder Adaptation (soundscape DAPT + supervised LoRA)

**Status**: smoke-gated; running the DAPT track (full runs in progress)
**Depends on**: exp 300 (extractor + ridge probe + unified cache), shared `birdclef.encoders`

## Execution decision (2026-06-11)

The adaptation we care about for the working note is **to the soundscape domain**, so
the **DAPT track is the prioritized run** this push (EAT + SSAMBA-HF). The reason it,
not LoRA, is the "fine-tune on soundscapes" path is a labeling constraint: the
soundscapes are **unlabeled except the 739 LOSO windows, which are the probe's own
evaluation set** — training supervised on them would leak the eval. So supervised
adaptation has to live on the focal recordings (LoRA track), while soundscape
adaptation has to be self-supervised (DAPT). Supervised soundscape adaptation via
teacher pseudo-labels (exp 010/101) is a possible follow-up, not in this push.

- **Now:** DAPT on the 10,658-file / ~177 h soundscape pool, `dapt-ep1` (EAT, SSAMBA-HF).
- **Deferred:** supervised-focal LoRA (BEATs/EAT/SSAMBA-HF), the LoRA rank sweep {16,32},
  and pseudo-label soundscape LoRA.

**Smoke gate (verified):** the LoRA smoke passed for all three encoders — `target_modules`
attach (BEATs 0.65 %, EAT 0.50 %, SSAMBA 2.27 % trainable) and the grad-enabled
`pooled_features` forward run — which also validates the shared forward DAPT reuses.
The first venv build exposed a missing dependency closure in `_setup-venv.sh` (the
shared `birdclef.__init__` needs `pandas`; EAT remote code needs `timm` +
`transformers<5`; the torch/torchaudio/torchvision trio must be pinned), now mirrored
from exp 300's proven stack.

**DAPT scale note:** `dapt-ep1` samples one 5 s window per 60 s file → ~666 optimizer
steps/epoch (~10–20 min/encoder on an RTX 6000), i.e. it sees ~8 % of the pool's audio.
That is the chosen primary; more windows/file or more epochs is the first knob if the Δ
comes out flat (cheap, given the runtime).

## Hypothesis

Adapting a foundational encoder to the bird/soundscape domain closes more of the
focal→soundscape gap than a frozen probe. Two adaptation axes, both reported as a
frozen→adapted **Δ macro AUROC** through exp 300's identical ridge LOSO probe:

1. **Supervised-focal LoRA** (BEATs, EAT, SSAMBA-HF) — parameter-efficient
   fine-tuning on focal weak labels (multi-label focal BCE head).
2. **Soundscape DAPT** (EAT, SSAMBA-HF) — continued self-supervised pretraining on
   the 176.5 h unlabeled Pantanal pool.

**BEATs is LoRA-only.** Its masked-prediction SSL objective needs a co-trained
discrete acoustic tokenizer; continued pretraining with a frozen tokenizer is a
known-degraded stretch, so it is deliberately out of scope (recorded honestly).

## Design — thin spoke over exp 300

301 produces adapted encoders; exp 300 does all extraction and probing:

```
lora/dapt → adapter dir | state_dict   →   birdclef-300 extract --variant <tag>
                                                 --adapter <dir> | --state-dict <pt>
                                       →   birdclef-300 probe --variant <tag>   →  ΔAUROC vs frozen
                                       →   birdclef-301 ksshift  (frozen vs adapted embeddings)
```

The adapted encoder is the **same** shared wrapper exp 300 extracts with, so the
adaptation and the comparison share preprocessing exactly (see `encoder_forward.py`,
which mirrors each wrapper's pooled forward but keeps autograd on — the wrappers'
`extract_features` is `@torch.no_grad()`).

## DAPT objective (genericity tradeoff, surfaced)

> **Superseded (2026-06-11):** the generic latent-bootstrap below was shelved as
> degenerate. The run uses each encoder's **native** objective instead — SSAMBA
> masked-spectrogram MPG/MPC (`ssamba_dapt.py`) and EAT data2vec-2.0 (`eat_dapt.py`,
> from-scratch PyTorch port). The latent-bootstrap is kept only as the "homemade
> objective" baseline. See `results.md`. The paragraph below documents the original
> tradeoff for the record.

DAPT uses a **data2vec / BYOL-style latent bootstrap**: a masked student predicts
the full-clip embedding of an EMA teacher, on target-domain audio only. This is one
generic, literature-grounded continued-pretraining objective that works identically
for EAT and SSAMBA-HF via the wrapper interface — rather than reconstructing each
encoder's exact original SSL loss. SSAMBA's exact masked-spectrogram-reconstruction
DAPT can alternatively reuse exp 103's pipeline; the tradeoff (uniform & simple vs.
encoder-faithful) is deliberate.

## Evaluation

- Primary: per-encoder frozen→adapted **Δ macro AUROC** (exp 300 probe on `frozen`
  vs `lora-r8` / `dapt-ep1` variants).
- Diagnostic: `ksshift` per-dim KS between frozen and adapted embeddings — how far
  adaptation moved the representation.
- Sweep: LoRA rank ∈ {8, 16, 32} (duplicate the config with a new `variant` tag).

## Out of scope

- End-to-end (jointly-trained head) evaluation — the ridge re-probe is the reported
  Δ so it is apples-to-apples with exp 300's frozen numbers.
- BEATs DAPT (frozen-tokenizer continued SSL).
