from __future__ import annotations

import numpy as np

from birdclef_107 import perch_features as pf


def test_project_perch_sidecar_maps_and_excludes_used_raw(monkeypatch):
    monkeypatch.setattr(pf, "get_species_list", lambda artifacts_dir=None: ["sp_a", "sp_b", "sp_c"])
    monkeypatch.setattr(
        pf,
        "load_species_mapping",
        lambda artifacts_dir=None: {
            "direct_map": {"sp_a": 1},
            "genus_proxy_map": {"sp_b": [2, 4]},
            "unmapped": ["sp_c"],
            "birdclef_species": ["sp_a", "sp_b", "sp_c"],
        },
    )

    logits = np.array(
        [[-3.0, 5.0, 1.0, 4.0, 2.0, 3.0]],
        dtype=np.float32,
    )
    out = pf.project_perch_sidecar(logits, topk=2)

    assert out["mapped_probs"].shape == (1, 3)
    np.testing.assert_allclose(
        out["mapped_probs"][0, 0],
        1.0 / (1.0 + np.exp(-5.0)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        out["mapped_probs"][0, 1],
        max(1.0 / (1.0 + np.exp(-1.0)), 1.0 / (1.0 + np.exp(-2.0))),
        rtol=1e-6,
    )
    assert out["mapped_probs"][0, 2] == 0.0

    # Raw indices 1,2,4 are already used by direct/genus-proxy mapping, so the
    # residual top-k must come from the remaining columns {0,3,5}.
    picked = set(out["topk_idx"][0].tolist())
    assert picked <= {0, 3, 5}
    assert 5 in picked  # highest remaining prob/logit
    assert 3 in picked  # second-highest remaining prob/logit
    assert np.all(out["topk_val"][0] > 0.0)
