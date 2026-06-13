"""Exp 301 — encoder adaptation (supervised-focal LoRA + soundscape DAPT).

Thin spoke over exp 300: trains adapted encoders, writes adapter dirs / DAPT
state_dicts, then re-drives `birdclef-300 extract --variant <tag>` + `probe` to
report the frozen→adapted Δ. See CLAUDE.md for the hub/spoke contract.
"""
