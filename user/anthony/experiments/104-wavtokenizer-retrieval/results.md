# Results: WavTokenizer Codec Retrieval for Bird Audio

**Date Completed**: 2026-03-31
**Author**: Anthony

## Summary

WavTokenizer's speech-trained codec destroys species-discriminative features in bird audio, failing both the round-trip quality gate and retrieval gate by wide margins.

**Verdict**: rejected

## Observations

### Quantitative

| Metric | Gate Threshold | Result | Verdict |
|--------|---------------|--------|---------|
| Perch embedding cosine similarity | > 0.90 | 0.313 mean | FAIL |
| Top-1 species match (round-trip) | > 80% | 9.0% | FAIL |
| Focal retrieval P@10 (best config) | > 0.80 | 0.195 (K=64, k=4) | FAIL |
| Soundscape recall@0.01 | positive signal | 0.002 | FAIL |

### Per-Species Round-Trip Quality (Perch v2)

| Species | Class | Cosine Sim | Top-1 Match | Top-5 Agree |
|---------|-------|-----------|-------------|-------------|
| Lesser Snouted Tree Frog | Amphibia | 0.539 | 25% | 22% |
| Domestic Dog | Mammalia | 0.423 | 10% | 16% |
| Pale-legged Weeping Frog | Amphibia | 0.374 | 5% | 9% |
| Purplish Jay | Aves | 0.357 | 5% | 17% |
| Common Potoo | Aves | 0.340 | 30% | 16% |
| Bare-faced Curassow | Aves | 0.337 | 15% | 20% |
| Rufous Hornero | Aves | 0.214 | 0% | 0% |
| Great Kiskadee | Aves | 0.201 | 0% | 3% |
| Chestnut-vented Conebill | Aves | 0.200 | 0% | 1% |
| White-faced Whistling Duck | Aves | 0.145 | 0% | 2% |

### Best Retrieval Config (K=64, k=4) Per-Species P@10

| Species | P@10 |
|---------|------|
| Purplish Jay | 0.360 |
| Bare-faced Curassow | 0.295 |
| Chestnut-vented Conebill | 0.275 |
| Domestic Dog | 0.270 |
| Pale-legged Weeping Frog | 0.200 |
| Great Kiskadee | 0.185 |
| Rufous Hornero | 0.120 |
| Lesser Snouted Tree Frog | 0.105 |
| Common Potoo | 0.070 |
| White-faced Whistling Duck | 0.065 |

### Qualitative

- Codec reconstruction sounds like "underwater" speech-like audio — bird call structure is not preserved
- The 10% random baseline for P@10 (10 species, uniform) means best retrieval at 0.195 is barely 2x random
- Soundscape search produces near-zero scores everywhere — motifs are not species-specific

## Analysis

See `analysis.md` for comprehensive breakdown.

## Artifacts

| Type | Location |
|------|----------|
| Logs | `logs/tokenize-6109361.out`, `logs/eval-6109751.out` |
| Round-trip scores | `artifacts/roundtrip_scores.parquet` |
| Retrieval results | `artifacts/retrieval_results.parquet` |
| Soundscape results | `artifacts/soundscape_results.parquet` |
| Codebook embeddings | `artifacts/codebook_embeddings.npy` |
| Token sequences | `artifacts/tokens/`, `artifacts/soundscape_tokens/` |
| Reconstructed audio | `artifacts/reconstructed/` |
| Species subset | `artifacts/subset.json` |

## Next Steps

- [ ] None planned — this direction is not viable with speech-trained codecs
- [ ] If revisiting: would need a codec fine-tuned on bioacoustic audio (not available off-the-shelf)
- [ ] The k-mer retrieval framework itself is sound — the bottleneck is tokenizer quality, not search

**Leads to**: No follow-up experiments. Confirms that off-the-shelf speech codecs don't transfer to bioacoustics. Focus remains on supervised approaches (exp 102) and SSL pretraining (exp 103).
