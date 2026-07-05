# Vendor directory

This directory is intentionally small. It contains:

- a few **git submodules** whose source is still imported by code or required by reproducibility scripts;
- `vendor/kaggle/`, which stores small vendored Kaggle notebook/dataset metadata snapshots tracked as normal files.

Most external repositories are documented under `docs/vendor/` and referenced by upstream URL instead of being checked out here.

## Retained submodules

Initialize only when you need one of these paths:

```bash
git submodule update --init vendor/<name>
```

| Path | Why retained |
|---|---|
| `vendor/unilm` | BEATs wrappers/scripts import `vendor/unilm/beats`. |
| `vendor/ssamba` | Original SSAMBA implementation and Vim dependency for SSAMBA verification/training paths. |
| `vendor/ssast` | SSAST wrapper imports the official implementation from `vendor/ssast/src`. |
| `vendor/WavTokenizer` | Experiment 104 loads configs and code from the upstream WavTokenizer checkout. |
| `vendor/sydorskyi-2025` | Parity oracle for the XC-pretraining/Sydorskyi ports. |
| `vendor/ESC-50` | Legacy encoder benchmark scripts can use the official ESC-50 checkout. |
| `vendor/bioacoustics-model-zoo` | Legacy/user editable installs can point at the local BMZ checkout. |
| `vendor/bambird` | Experiment 010 teacher-label pipeline can use editable bambird. |
| `vendor/scikit-maad` | Experiment 010 teacher-label pipeline can use editable scikit-maad. |

## Reference-only repositories

The following were removed as submodules. Keep using the docs in `docs/vendor/`; clone upstream manually into scratch/tmp if source inspection is needed.

| Former path | Upstream |
|---|---|
| `vendor/BirdNET-Analyzer` | https://github.com/birdnet-team/BirdNET-Analyzer |
| `vendor/BirdSet` | https://github.com/DBD-research-group/BirdSet |
| `vendor/EAT` | https://github.com/cwx-worst-one/EAT |
| `vendor/ProtoCLR` | https://github.com/ilyassmoummad/ProtoCLR |
| `vendor/Wav2vec2.0` | https://github.com/HarunoriKawano/Wav2vec2.0 |
| `vendor/ast` | https://github.com/YuanGongND/ast |
| `vendor/best-rq-pytorch` | https://github.com/lucasnewman/best-rq-pytorch |
| `vendor/biodenoising` | https://github.com/earthspecies/biodenoising |
| `vendor/biolingual` | https://github.com/david-rx/BioLingual |
| `vendor/birdclef-2025` | https://github.com/dsgt-arc/birdclef-2025 |
| `vendor/descript-audio-codec` | https://github.com/descriptinc/descript-audio-codec |
| `vendor/dinosr` | https://github.com/Alexander-H-Liu/dinosr |
| `vendor/flyingsquid` | https://github.com/HazyResearch/flyingsquid |
| `vendor/hubert` | https://github.com/bshall/hubert |
| `vendor/mamba-onnx` | https://github.com/acmiyaguchi/mamba-onnx |
| `vendor/model2vec` | https://github.com/MinishLab/model2vec |
| `vendor/opensoundscape` | https://github.com/kitzeslab/opensoundscape |
| `vendor/perch` | https://github.com/google-research/perch |
| `vendor/scaling-verification` | https://github.com/HazyResearch/scaling-verification |
