"""Perch + ProtoSSM pipeline for BirdCLEF 2026.

Implements the ProtoSSM v5 pipeline (0.925 LB):
  - Selective State Space Model (Mamba-style) for temporal modeling
  - Prototypical classification with gated Perch fusion
  - ResidualSSM second-pass correction
  - 6-stage post-processing pipeline
"""
