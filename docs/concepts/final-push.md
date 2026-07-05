# Concept: Final Push — Combined BirdCLEF 2026 Working Note

**Status**: historical paper-planning note
**Created**: 2026-06-08
**Updated**: 2026-06-09

## Purpose / scope

The competition closed on 2026-06-03, so the "final push" is the **write-up phase**, not more
leaderboard chasing. The goal is a single CEUR-WS working note for CLEF 2026.

**Central question.** *Can foundational audio encoders serve the BirdCLEF task, and how do they
compare with comparable systems from the Kaggle leaderboard?* The paper leads heavy with
audio-encoder benchmarking and SSAMBA architecture work, evaluates them on focal→soundscape transfer,
compares them head-to-head against the leaderboard's supervised CNN/SED and Perch-probe systems, and
closes with a discussion of what the **token-based paradigm** unlocks beyond raw score
(multimodality, sequence prompting, generative audio).

**Two bodies of work:**

- **Foundational-encoder research (primary, front-of-paper)** — drafted in the separate repo
  `/mnt/data/research/arc/8903-birdclef-2026/` (encoder benchmarks, SSAMBA/SSL, WavTokenizer codec,
  Perch transfer). Encoder-benchmark and SSAMBA-architecture material is the centerpiece.
- **The leaderboard systems (comparator)** — the team's 0.929-LB pipeline (HGNetV2-B0 SED + Perch-v2
  probes, score-fused) across `user/murilo/experiments/`, **not yet written up**. Serves as the
  yardstick the encoder approaches are compared against.

**Scope includes a pre-paper experiment phase.** The comparative thesis requires foundational
encoders measured *on the bird task* — today only Perch is. So exp **300** (cross-encoder
frozen-probe sweep + metric bridge) and exp **301** (supervised-focal LoRA + soundscape DAPT) must
run before writing; 300 supplies §6's comparison, 301 supplies §7's adaptation result and KS-shift
figure. Everything else is write-up of existing results. See *Experiments to run before writing*.

**Structure decision (deliberate).** An outline review recommended a system-paper spine with the
encoder material demoted (venue-optimal for LifeCLEF). We **override that on purpose**: foundational
encoders are the contribution and lead, heavy; the leaderboard system is the comparator, kept compact.

## Framing thesis

Lead question is **comparative and representational**: how far do foundational / token-based audio
encoders go on the bird task relative to specialized leaderboard systems?

The most interesting finding is the **nuance**: the leaderboard frontier *already* rests on a
foundational encoder — Perch-v2 probing reaches 0.925 LB and the 0.929 ensemble leans on it. So this
is not "foundational encoders vs. the leaderboard" as rival camps. It is:

- **Bioacoustic-specialist foundational encoder (Perch) + linear probe** → already leaderboard-competitive.
- **General / token-based foundational encoders (BEATs, AST, EAT, SSAMBA) + neural codec (WavTokenizer)**
  → strong on proxy tasks, not yet matching specialized supervised systems on the bird task.
- **The forward case** → token-based encoders, even when they don't top the score, offer capabilities
  discriminative CNNs cannot (below).

Focal→soundscape **shift** is the setting that stresses transfer (domain: 96.8% of Perch logits shift,
median KS 0.177; coverage: 159/234 species with zero labeled soundscape; validation: offline metrics
anti-correlate with LB). **Metric-bridging is the crux of "how do they compare"** — ESC-50 accuracy,
HSN cmAP, LOSO AUROC, and Kaggle LB are not directly comparable, and reconciling them is a central
methodological point, not a footnote.

## Working-note outline & status

Encoder benchmarks (§3) and SSAMBA (§4) are the heavy, primary sections. §6 is the comparison against
leaderboard systems; §7 is the token-paradigm discussion + future work. Pointers: drafted prose lives
in the 8903 repo (`sections/**/*.tex`); experiment numbers resolve under
`user/{anthony,murilo}/experiments/` (see Connections).

| § | Section | Sources | Headline | Status |
|---|---------|---------|----------|--------|
| 1 | Introduction — comparative + token-paradigm framing | `00_main.tex` | foundational encoders for the bird task; how they compare; what tokens unlock | mostly drafted (intro is research-questions style — re-point to the comparison) |
| 2 | Data & shift setting | `00_main.tex` (data) + `106`/`107` | asymmetry table; 206→234; 159/234 zero-soundscape; Perch KS-shift 96.8% | data table drafted |
| 3 | **Encoder benchmarks (HEAVY, primary)** | `001`/`002`/`003` → `10_benchmarks.tex` | BEATs 95.8% / AST 95.5% ESC-50; EAT LoRA 77.9→94.6; static-codebook 17× CPU at −48 pts; pooler ladder; distillation | drafted — **expand** (full encoder matrix, probe-vs-LoRA, Model2Vec deployment) |
| 4 | **SSAMBA: architecture, scaling & SSL (HEAVY, primary)** | `004`–`008`, `103` → `20_ssamba.tex` | exact HF port (cos 1.000); patch-geometry sweep + O(n) regime ~6k tok; ONNX selective-scan op; AudioSet→HSN 0.179 cmAP; SSL at 55k inconclusive | drafted — **expand** (token geometry, CPU kernel, full SSL arc) |
| 5 | Perch transfer & shift characterization | `105`/`106`/`107` → `40_perch.tex` | Hopfield 0.761 / ridge 0.749 AUROC; focal-vs-no-focal cliff; insects unreachable | drafted |
| 6 | Comparison with leaderboard systems (comparator) | **`300`** + `011`/`012`/`016`–`023`/`035`/`010` + `024`/`025`/`033`/`037` | cross-encoder bird-task probes (BEATs/AST/EAT/SSAMBA/BirdNET) vs. supervised SED v5j 0.906/0.917, Perch V18 0.925, fusion **0.929 LB**; pseudo-labeling 0/5; **metric-bridge** | blocked on 300 — then write; keep compact |
| 7 | Discussion — the token-based paradigm + future work | `104` → `30_wavtokenizer.tex`; **`301`** | multimodality · sequence prompting · generative audio; WavTokenizer as generative-token probe; **301** adaptation Δ + KS-shift figure | blocked on 301 — then write |

### Detail

- **Encoder benchmarks (§3, heavy).** Full encoder matrix (BEATs, AST, EAT, SSAMBA, wav2vec2) under
  linear-probe and LoRA; the BEATs deployment / Model2Vec static-codebook study (17× CPU speedup, the
  AttentionPooler→CNN1DPooler recovery ladder) as a first-class subsection; the BEATs→ConvNeXt1D
  distillation attempt and why it was dropped.
- **SSAMBA (§4, heavy).** HF/PyTorch port (cosine-exact); patch-shape / token-geometry sweep and the
  O(n) selective-scan regime; the ONNX custom selective-scan operator for CPU; AudioSet→HSN transfer
  conditions; the BirdCLEF 55k SSL run and its scale limitation. Architecture + engineering depth is
  the contribution, and SSAMBA most embodies the "token sequence" thesis (semantic tokens over a
  soundscape, the prompting analogy).
- **Comparison (§6).** Present the leaderboard systems compactly as the comparator — supervised CNN/SED
  *and* the Perch-probe ensemble — then a comparison that confronts metric-bridging head-on (how an
  encoder probe's LOSO AUROC relates to a system's Kaggle LB). The pseudo-labeling 0/5 result belongs
  here as the comparator system's attempt to exploit unlabeled data, and why it stalled.
- **Discussion — token paradigm (§7).** The forward case for token/foundational encoders beyond score:
  - **Multimodality** — encoder embeddings align audio with text/image (CLAP-style); a token system
    could ingest taxonomy, field notes, or images, unlike a closed-vocabulary CNN classifier.
  - **Sequence prompting** — SSAMBA's hypothesis: contextualize a soundscape like an LLM prompt, with
    in-context conditioning on site/season/few-shot exemplars without retraining — a token sequence
    enables this; a pooled CNN logit cannot.
  - **Generative audio** — neural audio codecs (WavTokenizer) are the tokenizer class behind generative
    audio models; synthesis could attack the 159 zero-soundscape coverage gap. The WavTokenizer result
    is the first honest probe of whether codec token spaces carry bioacoustic semantics (they don't yet
    for high-frequency birds) — grounding the vision rather than overselling it.
  - **Genuine future work (post-paper)** — extend the cross-encoder sweep to more encoders, generative
    synthesis to attack the 159 zero-soundscape coverage gap, and multimodal (audio+text/taxonomy)
    probing. (The 300/301 experiments themselves are now *pre-paper requirements*, not future work —
    see below.)

## Experiments to run before writing

Two experiment programs gate the paper; everything else is write-up of existing results. (Planning
only here — folders not yet scaffolded.)

### 300 — cross-encoder transfer (frozen-probe comparison) → feeds §6
Place every foundational encoder on the *same* BirdCLEF 2026 yardstick as Perch and the leaderboard.
- Extract 5s-window embeddings (749 labeled windows + focal prototypes) for **BEATs, AST, EAT,
  SSAMBA, BirdNET**; mean-pool to a fixed vector; write a common embedding-cache schema. Reuse
  `106` `extract.py` / `data_loading.py`; Luigi-sharded extraction (6 encoders × many files).
- Run each through the **`106` ridge + `107` Hopfield heads** on the 9-fold LOSO split; report
  cMAP + AUROC with the birds/amphibians/insects and with/without-focal breakdown.
- **Metric bridge**: Perch has both LOSO AUROC and Kaggle LB — use it to calibrate AUROC↔LB so the
  probes sit on the leaderboard axis. Optional: one late Kaggle sub of the best probe for a real
  non-Perch LB anchor (if late submissions are open).

### 301 — encoder adaptation (supervised LoRA + soundscape DAPT) → feeds §7
Test whether adapting the encoder closes the focal→soundscape gap better than a frozen probe.
- Pick 2–3 encoders (**BEATs, EAT, SSAMBA**); two tracks sharing one LoRA/encoder-wrapper stack:
  supervised-focal LoRA (rank 8/16/32 sweep) on focal weak labels, and **soundscape DAPT** —
  continue each encoder's SSL objective on the 176.5h unlabeled Pantanal pool (EAT/SSAMBA tractable;
  BEATs a stretch via frozen tokenizer).
- Re-extract embeddings, re-run the same 300 heads. Report frozen→adapted Δ per encoder and the
  `106` **KS-shift before/after** (does DAPT pull the distributions together?).

**Sequencing & folders.** 300 first (cheap, encoder-agnostic, establishes the comparison and the
cache schema), then 301 (heavy, reuses 300's caches + heads). Hub-and-spoke:
`300-cross-encoder-transfer` (eval hub) + `301-encoder-adaptation` (two-track spoke), decoupled by
the embedding-cache schema — no cross-experiment Python imports (PACE per-experiment venvs).

## Open questions

- **Venue tradeoff (acknowledged).** Foregrounding encoder/architecture work optimizes for the author's
  individual research contribution over LifeCLEF system-paper convention, and de-emphasizes the 0.929
  result. Deliberate — recorded so it stays a conscious choice.
- **Negatives placement.** With the comparative framing, the three negatives now have natural homes
  (SSAMBA-SSL in §4, pseudo-labeling in §6, WavTokenizer in §7's generative thread) rather than one
  consolidated section — the earlier "consolidate negatives" note is superseded.
- **Metric reconciliation (now central).** Needs an explicit treatment in §6: a table or paragraph
  mapping ESC-50 acc / cmAP / LOSO AUROC / Kaggle LB and stating what can and cannot be compared.
- **System-block framing.** §6 reads as "the team's systems, compared against," not a co-equal authored
  half — avoids over-claiming the 0.929 pipeline. Framing only (no who-did-what enumeration).
- **Deadline.** Confirm the exact CLEF 2026 working-notes / CEUR-WS submission date.
- **Reproducibility.** How much system work lives only on feature branches (`mg/*`) vs. `main`.
- **Stale docs.** Whether to refresh `STATUS.md` / `STRATEGY.md` (last touched 2026-04-20) alongside.

## Connections

- [[STATUS]], [[STRATEGY]] — project state and two-track strategy (both stale).
- Foundational-encoder draft: `/mnt/data/research/arc/8903-birdclef-2026/` (LaTeX, CEUR-WS).
- Encoder/representation experiments: `user/anthony/experiments/{001..107}/`.
- Leaderboard-system experiments: `user/murilo/experiments/{010..037}/`.
- Pre-paper experiments (required, not yet scaffolded): `300` cross-encoder transfer, `301` encoder
  adaptation / DAPT.
