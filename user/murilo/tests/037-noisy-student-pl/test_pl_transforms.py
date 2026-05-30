"""Exp 037 — soft-PL transform tests.

The whole experiment hinges on NOT repeating exp 033's sharpening mistake. These tests pin
the transform semantics:
  - temperature_scale(p, T) with T>1 SOFTENS (pulls toward 0.5), is monotonic, stays in [0,1].
  - power_transform(p, gamma) with gamma<1 RAISES low probs, is monotonic, stays in [0,1].
  - identity at T=1 / gamma=1.
  - sigmoid/to_logit roundtrip.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parent.parent.parent / "experiments/037-noisy-student-pl/scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load("generate_ensemble_pl")


def test_temperature_identity():
    p = np.array([0.01, 0.2, 0.5, 0.8, 0.99], dtype=np.float64)
    np.testing.assert_allclose(gen.temperature_scale(p, 1.0), p, atol=1e-12)


def test_temperature_softens_toward_half():
    p = np.array([0.05, 0.3, 0.5, 0.7, 0.95], dtype=np.float64)
    out = gen.temperature_scale(p, 2.0)
    # values above 0.5 move down toward 0.5; below 0.5 move up toward 0.5; 0.5 stays
    assert out[0] > p[0] and out[1] > p[1]
    assert abs(out[2] - 0.5) < 1e-9
    assert out[3] < p[3] and out[4] < p[4]
    # never crosses 0.5 (monotone, confidence reduced not flipped)
    assert (out[:2] < 0.5).all() and (out[3:] > 0.5).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_temperature_monotonic():
    p = np.linspace(0.001, 0.999, 50)
    out = gen.temperature_scale(p, 2.5)
    assert np.all(np.diff(out) > 0)


def test_power_identity():
    p = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float64)
    np.testing.assert_allclose(gen.power_transform(p, 1.0), p, atol=1e-12)


def test_power_gamma_lt1_raises():
    p = np.array([0.01, 0.25, 0.5, 0.81], dtype=np.float64)
    out = gen.power_transform(p, 0.5)
    # p^0.5 >= p for p in [0,1]; equality only at 0 and 1
    assert np.all(out >= p - 1e-12)
    assert out[1] > p[1]  # 0.25^0.5 = 0.5
    assert abs(out[3] - 0.9) < 1e-9  # 0.81^0.5 = 0.9
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_power_monotonic():
    p = np.linspace(0.0, 1.0, 50)
    out = gen.power_transform(p, 0.5)
    assert np.all(np.diff(out) >= -1e-12)


def test_sigmoid_logit_roundtrip():
    p = np.array([0.02, 0.3, 0.5, 0.77, 0.98], dtype=np.float64)
    np.testing.assert_allclose(gen.sigmoid(gen.to_logit(p)), p, atol=1e-6)
