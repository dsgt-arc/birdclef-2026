import numpy as np
import pytest
import soundfile as sf

from birdclef_kaggle.audio import (
    SAMPLES_PER_CLIP,
    SAMPLES_PER_WINDOW,
    SAMPLE_RATE,
    WINDOWS_PER_CLIP,
    load_clip,
    stack_windows,
)


def _write_sine(path, seconds, freq=440, sr=SAMPLE_RATE):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, np.sin(2 * np.pi * freq * t).astype(np.float32), sr)


def test_load_exact_60s(tmp_path):
    p = tmp_path / "a.wav"
    _write_sine(p, 60)
    clip = load_clip(p)
    assert clip.windows.shape == (WINDOWS_PER_CLIP, SAMPLES_PER_WINDOW)
    assert clip.end_seconds == tuple(range(5, 65, 5))


def test_load_short_clip_is_zero_padded(tmp_path):
    p = tmp_path / "short.wav"
    _write_sine(p, 30)
    clip = load_clip(p)
    assert clip.windows.shape == (WINDOWS_PER_CLIP, SAMPLES_PER_WINDOW)
    # windows 7..12 (30-60s) should be zero-padded
    assert np.allclose(clip.windows[6:], 0.0)


def test_load_long_clip_is_truncated(tmp_path):
    p = tmp_path / "long.wav"
    _write_sine(p, 90)
    clip = load_clip(p)
    assert clip.windows.shape == (WINDOWS_PER_CLIP, SAMPLES_PER_WINDOW)


def test_load_stereo_is_averaged(tmp_path):
    p = tmp_path / "stereo.wav"
    n = SAMPLES_PER_CLIP
    left = np.ones(n, dtype=np.float32)
    right = -np.ones(n, dtype=np.float32)
    sf.write(p, np.stack([left, right], axis=1), SAMPLE_RATE, subtype="FLOAT")
    clip = load_clip(p)
    assert np.allclose(clip.windows, 0.0)


def test_load_wrong_sample_rate_raises(tmp_path):
    p = tmp_path / "wrong_sr.wav"
    n = 16_000 * 60  # 60s at 16 kHz
    sf.write(p, np.zeros(n, dtype=np.float32), 16_000)
    with pytest.raises(ValueError, match="Expected sr"):
        load_clip(p)


def test_stack_windows_concatenates(tmp_path):
    p1 = tmp_path / "a.wav"
    p2 = tmp_path / "b.wav"
    _write_sine(p1, 60)
    _write_sine(p2, 60)
    clips = [load_clip(p1), load_clip(p2)]
    stacked = stack_windows(clips)
    assert stacked.shape == (2 * WINDOWS_PER_CLIP, SAMPLES_PER_WINDOW)


def test_stack_windows_empty():
    stacked = stack_windows([])
    assert stacked.shape == (0, SAMPLES_PER_WINDOW)
