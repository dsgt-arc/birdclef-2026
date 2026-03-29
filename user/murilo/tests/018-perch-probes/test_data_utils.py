"""Tests for data reshaping and metadata utilities.

Cross-ref: ref_yuriy_protossm_v5.py lines 1598-1658.
"""

import numpy as np
import pandas as pd
import pytest

from birdclef.perch.data_utils import (
    build_file_families,
    build_site_mapping,
    build_taxonomy_groups,
    get_file_metadata,
    reshape_to_files,
)


class TestReshapeToFiles:
    def test_basic_reshape(self):
        flat = np.arange(24 * 5).reshape(24, 5).astype(np.float32)
        meta = pd.DataFrame({"filename": ["f1"] * 12 + ["f2"] * 12})
        files, file_list = reshape_to_files(flat, meta)
        assert files.shape == (2, 12, 5)
        assert file_list == ["f1", "f2"]

    def test_preserves_values(self):
        flat = np.arange(12).reshape(12, 1).astype(np.float32)
        meta = pd.DataFrame({"filename": ["f1"] * 12})
        files, _ = reshape_to_files(flat, meta)
        np.testing.assert_array_equal(files.reshape(-1, 1), flat)

    def test_file_order_preserved(self):
        """File order should match first occurrence in meta_df."""
        meta = pd.DataFrame({"filename": ["b"] * 12 + ["a"] * 12})
        flat = np.zeros((24, 1), dtype=np.float32)
        _, file_list = reshape_to_files(flat, meta)
        assert file_list == ["b", "a"]  # order of first occurrence

    def test_mismatch_raises(self):
        flat = np.zeros((10, 5), dtype=np.float32)
        meta = pd.DataFrame({"filename": ["f1"] * 10})
        with pytest.raises(AssertionError):
            reshape_to_files(flat, meta, n_windows=12)

    def test_1d_features(self):
        flat = np.arange(24).astype(np.float32)
        meta = pd.DataFrame({"filename": ["f1"] * 12 + ["f2"] * 12})
        # Can't reshape 1D — need 2D input
        flat_2d = flat.reshape(-1, 1)
        files, _ = reshape_to_files(flat_2d, meta)
        assert files.shape == (2, 12, 1)


class TestBuildSiteMapping:
    def test_basic_mapping(self):
        meta = pd.DataFrame({"site": ["S1", "S1", "S2", "S2", "S3", "S3"]})
        mapping, n = build_site_mapping(meta, n_sites_max=20)
        assert n == 3
        assert set(mapping.keys()) == {"S1", "S2", "S3"}

    def test_capped_at_max(self):
        meta = pd.DataFrame({"site": [f"S{i}" for i in range(30)]})
        mapping, n = build_site_mapping(meta, n_sites_max=10)
        assert n == 10
        # Values should be capped at n_sites_max - 1
        assert all(v <= 9 for v in mapping.values())

    def test_missing_site_column(self):
        meta = pd.DataFrame({"filename": ["f1", "f2"]})
        mapping, n = build_site_mapping(meta)
        assert n == 0
        assert mapping == {}


class TestGetFileMetadata:
    def test_basic(self):
        meta = pd.DataFrame(
            {
                "filename": ["f1"] * 12 + ["f2"] * 12,
                "site": ["S1"] * 12 + ["S2"] * 12,
                "hour": [6] * 12 + [18] * 12,
            }
        )
        site_to_idx = {"S1": 0, "S2": 1}
        site_ids, hour_ids = get_file_metadata(meta, ["f1", "f2"], site_to_idx)
        np.testing.assert_array_equal(site_ids, [0, 1])
        np.testing.assert_array_equal(hour_ids, [6, 18])

    def test_unknown_site(self):
        meta = pd.DataFrame(
            {
                "filename": ["f1"] * 12,
                "site": ["UNKNOWN"] * 12,
                "hour": [10] * 12,
            }
        )
        site_to_idx = {"S1": 0}
        site_ids, hour_ids = get_file_metadata(meta, ["f1"], site_to_idx)
        assert site_ids[0] == 0  # defaults to 0 for unknown


class TestBuildTaxonomyGroups:
    def test_basic(self):
        taxonomy_df = pd.DataFrame(
            {
                "primary_label": ["sp1", "sp2", "sp3"],
                "scientific_name": ["Genus1 species1", "Genus1 species2", "Genus2 species3"],
            }
        )
        n_fam, c2f, f2i = build_taxonomy_groups(taxonomy_df, ["sp1", "sp2", "sp3"])
        assert n_fam == 2  # Genus1, Genus2
        assert c2f[0] == c2f[1]  # sp1 and sp2 share genus
        assert c2f[2] != c2f[0]


class TestBuildFileFamilies:
    def test_basic(self):
        labels = np.zeros((2, 12, 5), dtype=np.float32)
        labels[0, 0, 0] = 1.0  # file 0, class 0
        labels[1, 3, 2] = 1.0  # file 1, class 2
        class_to_family = [0, 0, 1, 1, 2]
        fam = build_file_families(labels, class_to_family, n_families=3)
        assert fam.shape == (2, 3)
        assert fam[0, 0] == 1.0  # family 0 (from class 0)
        assert fam[1, 1] == 1.0  # family 1 (from class 2)
        assert fam[0, 2] == 0.0  # file 0 has no family 2
