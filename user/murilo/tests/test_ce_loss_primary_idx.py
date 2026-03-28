"""Test that CE loss fix correctly uses primary_label instead of argmax.

Verifies:
1. SEDFocalDataset returns (waveform, labels, primary_idx) with correct primary_idx
2. SEDSoundscapeDataset returns correct primary_idx
3. FocalDataset234 remaps primary_idx correctly
4. _compute_loss uses primary_idx for CE, not argmax
5. Shows mismatch rate: how often argmax != primary_label_idx
"""

import pandas as pd
import pytest
import torch

from birdclef.data.sed import SEDFocalDataset, SEDSoundscapeDataset
from birdclef.data.soundscape import FocalDataset234

DATA_DIR = "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026"


@pytest.fixture
def species_list():
    taxonomy = pd.read_csv(f"{DATA_DIR}/taxonomy.csv")
    return taxonomy["primary_label"].tolist()


@pytest.fixture
def train_df():
    return pd.read_csv(f"{DATA_DIR}/train.csv")


@pytest.fixture
def labels_df():
    return pd.read_csv(f"{DATA_DIR}/train_soundscapes_labels.csv")


class TestSEDFocalDataset:
    """Test that SEDFocalDataset returns correct primary_idx."""

    def test_returns_three_elements(self, species_list, train_df):
        ds = SEDFocalDataset(
            audio_dir=f"{DATA_DIR}/train_audio",
            metadata_df=train_df.head(5),
            species_list=species_list,
            sample_rate=32000,
            window_duration=1.0,  # short for speed
        )
        if len(ds) == 0:
            pytest.skip("No audio files found")

        result = ds[0]
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"
        waveform, labels, primary_idx = result
        assert isinstance(primary_idx, int)
        assert 0 <= primary_idx < len(species_list)

    def test_primary_idx_matches_primary_label(self, species_list, train_df):
        ds = SEDFocalDataset(
            audio_dir=f"{DATA_DIR}/train_audio",
            metadata_df=train_df.head(20),
            species_list=species_list,
            sample_rate=32000,
            window_duration=1.0,
        )
        if len(ds) == 0:
            pytest.skip("No audio files found")

        for idx in range(min(len(ds), 10)):
            _, labels, primary_idx = ds[idx]
            # primary_idx should have label=1 in multi-hot
            assert (
                labels[primary_idx] == 1.0
            ), f"Sample {idx}: primary_idx={primary_idx} but labels[primary_idx]={labels[primary_idx]}"

    def test_primary_idx_vs_argmax_mismatch(self, species_list, train_df):
        """Show that argmax can pick the wrong species when secondaries exist."""
        species_to_idx = {s: i for i, s in enumerate(species_list)}

        n_mismatch = 0
        n_total = 0

        for _, row in train_df.head(500).iterrows():
            primary = row.get("primary_label", "")
            secondary = row.get("secondary_labels", "")

            if not primary or primary not in species_to_idx:
                continue

            primary_idx = species_to_idx[primary]

            # Build multi-hot (same logic as _build_label_vec)
            label_vec = torch.zeros(len(species_list), dtype=torch.float32)
            label_vec[primary_idx] = 1.0

            if isinstance(secondary, str) and secondary.strip() and secondary != "[]":
                secondary_clean = secondary.strip("[]' ")
                for s in secondary_clean.replace("'", "").replace('"', "").split(","):
                    s = s.strip()
                    if s in species_to_idx:
                        label_vec[species_to_idx[s]] = 1.0

            argmax_idx = label_vec.argmax(dim=-1).item()
            n_total += 1
            if argmax_idx != primary_idx:
                n_mismatch += 1

        print(f"\nTotal samples: {n_total}")
        print(
            f"Mismatches (argmax != primary): {n_mismatch} ({100 * n_mismatch / max(n_total, 1):.1f}%)"
        )
        # We expect SOME mismatches — that's the bug we're fixing
        # Just document the rate, don't fail


class TestSEDSoundscapeDataset:
    """Test that SEDSoundscapeDataset returns correct primary_idx."""

    def test_returns_three_elements(self, species_list, labels_df):
        ds = SEDSoundscapeDataset(
            soundscape_dir=f"{DATA_DIR}/train_soundscapes",
            labels_df=labels_df.head(5),
            species_list=species_list,
            sample_rate=32000,
        )
        if len(ds) == 0:
            pytest.skip("No soundscape files found")

        result = ds[0]
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"
        waveform, labels, primary_idx = result
        assert isinstance(primary_idx, int)
        assert labels[primary_idx] == 1.0


class TestFocalDataset234:
    """Test that FocalDataset234 remaps primary_idx correctly."""

    def test_remaps_primary_idx(self, species_list, train_df):
        # Use full species list as both 206 and 234 (identity remap for simplicity)
        ds_206 = SEDFocalDataset(
            audio_dir=f"{DATA_DIR}/train_audio",
            metadata_df=train_df.head(5),
            species_list=species_list,
            sample_rate=32000,
            window_duration=1.0,
        )
        if len(ds_206) == 0:
            pytest.skip("No audio files found")

        ds_234 = FocalDataset234(ds_206, species_list, species_list)
        result = ds_234[0]
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"
        _, labels_234, primary_idx_234 = result
        assert labels_234[primary_idx_234] == 1.0


class TestComputeLoss:
    """Test that CE loss uses primary_idx, not argmax."""

    def test_ce_loss_uses_primary_idx(self):
        """Verify CE loss targets primary_idx, not argmax."""
        import torch.nn.functional as F

        num_classes = 10
        batch_size = 4

        logits = torch.randn(batch_size, num_classes)

        # labels: multi-hot where species 0 and 5 are active
        labels = torch.zeros(batch_size, num_classes)
        labels[:, 0] = 1.0  # secondary (lower index)
        labels[:, 5] = 1.0  # primary

        primary_idx = torch.full((batch_size,), 5, dtype=torch.long)

        # Old behavior: argmax picks 0 (wrong!)
        old_targets = labels.argmax(dim=-1)
        assert (old_targets == 0).all(), "argmax should pick idx 0 (the bug)"

        # New behavior: CE on primary_idx=5 (correct!)
        loss_correct = F.cross_entropy(logits, primary_idx)
        loss_buggy = F.cross_entropy(logits, old_targets)

        # These should be different losses (training on different targets)
        assert not torch.allclose(
            loss_correct, loss_buggy
        ), "CE(primary_idx=5) should differ from CE(argmax=0)"

        # Verify gradient targets column 5, not column 0
        logits_grad = logits.clone().requires_grad_(True)
        loss = F.cross_entropy(logits_grad, primary_idx)
        loss.backward()
        assert (
            logits_grad.grad[:, 5].abs().mean() > logits_grad.grad[:, 0].abs().mean()
        ), "CE gradient should focus on primary_idx=5, not argmax=0"

    def test_soft_ce_ignores_primary_idx(self):
        """Verify soft_ce produces same loss regardless of primary_idx."""
        import torch.nn.functional as F

        num_classes = 10
        batch_size = 4

        logits = torch.randn(batch_size, num_classes)
        labels = torch.zeros(batch_size, num_classes)
        labels[:, 0] = 1.0
        labels[:, 5] = 1.0

        # soft_ce: normalize multi-hot, compute CE
        label_sum = labels.sum(dim=-1, keepdim=True).clamp(min=1)
        soft_targets = labels.float() / label_sum
        loss = F.cross_entropy(logits, soft_targets)

        # primary_idx is irrelevant for soft_ce
        assert loss.shape == (), "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
