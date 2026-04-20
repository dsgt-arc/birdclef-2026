import numpy as np

from birdclef_kaggle.ood import neutral_site_score, resolve_site_indices


def test_resolve_site_indices_all_known():
    idx, known = resolve_site_indices(["S01", "S22", "S23"])
    np.testing.assert_array_equal(idx, [0, 21, 22])
    np.testing.assert_array_equal(known, [True, True, True])


def test_resolve_site_indices_mixed():
    idx, known = resolve_site_indices(["S01", "S99", "S12"])
    assert idx[0] == 0
    assert idx[1] == -1
    assert idx[2] == 11
    np.testing.assert_array_equal(known, [True, False, True])


def test_resolve_site_indices_empty():
    idx, known = resolve_site_indices([])
    assert idx.shape == (0,)
    assert known.shape == (0,)


def test_neutral_site_score_shape_and_value():
    s = neutral_site_score((3, 234))
    assert s.shape == (3, 234)
    assert s.dtype == np.float32
    assert (s == 1.0).all()


def test_neutral_site_score_broadcasts_as_one():
    logits = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    site_score = neutral_site_score(logits.shape)
    result = logits * site_score
    np.testing.assert_allclose(result, logits)
