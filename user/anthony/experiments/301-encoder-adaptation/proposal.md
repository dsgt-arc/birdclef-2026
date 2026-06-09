# Experiment 301: Encoder Adaptation (supervised LoRA + soundscape DAPT)

**Status**: scaffolded (not yet run)
**Depends on**: exp 300 (extractor + ridge probe + unified cache), shared `birdclef.encoders`

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
