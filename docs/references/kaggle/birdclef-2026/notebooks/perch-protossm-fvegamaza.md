---
slug: "fvegamaza/birdclef-2026-perch-protossm"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-24"
votes: 0
---

# Kaggle Summary: BirdCLEF 2026 Perch + ProtoSSM (fvegamaza)

**Source**: https://www.kaggle.com/code/fvegamaza/birdclef-2026-perch-protossm
**Competition**: BirdCLEF+ 2026
**Score**: 0.913 LB

## Summary

Foundation model pipeline using Perch v2 + ProtoSSM temporal modeling + MLP probes +
metadata priors. **Functionally identical to zirach's notebook** — see below.

## Identity with zirach's Notebook

After downloading and comparing both notebooks, they are **byte-for-byte identical**
except for Unicode formatting in comments (em dash `–` vs ASCII hyphen `-`, 19 instances).

- Same 42 code cells, same structure
- Same hyperparameters, architecture, training loop, inference pipeline
- Same artifact paths, dataset sources, configuration values
- Neither acknowledges the other

This is not a fork with modifications — the code is identical. Both achieve 0.913 LB.

## Full Technical Details

See [perch-protossm-zirach.md](perch-protossm-zirach.md) for the complete technical
breakdown. Everything documented there applies identically to this notebook.

## Connections

- **Identical to**: [perch-protossm-zirach](perch-protossm-zirach.md)
- **Same family**: [perch-blamerx](perch-blamerx.md), [pantanal-distill](pantanal-distill.md)
