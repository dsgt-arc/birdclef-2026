"""Tests for nnAudio pipeline integration (DataModule + FrameTokenizedViT).

Validates that:
1. _WaveformDataset returns correct shapes (raw waveforms + labels)
2. FrameTokenizedViT end-to-end: waveform in -> logits out
3. GPU mel computation works inside model forward pass
4. All 4 patch configurations produce correct output shapes
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

# Add shared src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdset_benchmark.lightning.birdset_datamodule import _CachedWaveformDataset
from birdset_benchmark.models.frame_vit import FrameTokenizedViT

SAMPLE_RATE = 32000
DURATION = 5.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
NUM_CLASSES = 21  # HSN dataset


# ---------------------------------------------------------------------------
# Mock dataset (avoids HuggingFace download in tests)
# ---------------------------------------------------------------------------


class _MockBirdSetDataset(Dataset):
    """Mimics BirdSetDataset.__getitem__ return format with synthetic data."""

    def __init__(self, num_samples: int = 16, num_classes: int = NUM_CLASSES):
        self.num_samples = num_samples
        self.num_classes = num_classes
        torch.manual_seed(42)
        self.waveforms = torch.randn(num_samples, N_SAMPLES)
        self.labels = torch.zeros(num_samples, num_classes)
        # Assign 1-2 random labels per sample
        for i in range(num_samples):
            idx = torch.randint(0, num_classes, (2,))
            self.labels[i, idx] = 1.0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "audio": self.waveforms[idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dataset():
    return _MockBirdSetDataset(num_samples=8)


@pytest.fixture
def waveform_dataset(mock_dataset):
    """Simulate what _precompute_waveforms produces: pre-loaded tensors."""
    waveforms = mock_dataset.waveforms  # (N, num_samples)
    labels = mock_dataset.labels  # (N, num_classes)
    return _CachedWaveformDataset(waveforms, labels)


@pytest.fixture
def batch_waveform():
    torch.manual_seed(42)
    return torch.randn(4, N_SAMPLES)


# ---------------------------------------------------------------------------
# _WaveformDataset tests
# ---------------------------------------------------------------------------


class TestWaveformDataset:
    def test_length(self, waveform_dataset, mock_dataset):
        assert len(waveform_dataset) == len(mock_dataset)

    def test_returns_tuple(self, waveform_dataset):
        item = waveform_dataset[0]
        assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
        assert len(item) == 2, f"Expected 2-tuple (audio, labels), got {len(item)}"

    def test_audio_shape(self, waveform_dataset):
        audio, _ = waveform_dataset[0]
        assert audio.shape == (N_SAMPLES,), f"Expected ({N_SAMPLES},), got {audio.shape}"

    def test_labels_shape(self, waveform_dataset):
        _, labels = waveform_dataset[0]
        assert labels.shape == (NUM_CLASSES,), f"Expected ({NUM_CLASSES},), got {labels.shape}"

    def test_audio_dtype(self, waveform_dataset):
        audio, _ = waveform_dataset[0]
        assert audio.dtype == torch.float32

    def test_labels_dtype(self, waveform_dataset):
        _, labels = waveform_dataset[0]
        assert labels.dtype == torch.float32

    def test_labels_are_multi_hot(self, waveform_dataset):
        """Labels should only contain 0.0 or 1.0."""
        _, labels = waveform_dataset[0]
        assert ((labels == 0.0) | (labels == 1.0)).all()

    def test_dataloader_batching(self, waveform_dataset):
        """Verify waveforms can be collated into batches by DataLoader."""
        from torch.utils.data import DataLoader

        loader = DataLoader(waveform_dataset, batch_size=4, shuffle=False)
        batch_audio, batch_labels = next(iter(loader))
        assert batch_audio.shape == (4, N_SAMPLES)
        assert batch_labels.shape == (4, NUM_CLASSES)


# ---------------------------------------------------------------------------
# FrameTokenizedViT end-to-end tests (CPU)
# ---------------------------------------------------------------------------

# Patch configs: (patch_size, expected_tokens)
PATCH_CONFIGS = [
    pytest.param((16, 16), 248, id="baseline_16x16"),
    pytest.param((128, 2), 250, id="vertical_128x2"),
    pytest.param((128, 4), 125, id="vertical_128x4"),
    pytest.param((128, 1), 500, id="vertical_128x1"),
]


class TestFrameTokenizedViT:
    def test_forward_baseline(self, batch_waveform):
        """Baseline 16x16: waveform -> logits with correct shape."""
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
            patch_size=(16, 16),
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch_waveform)
        assert logits.shape == (4, NUM_CLASSES), f"Expected (4, {NUM_CLASSES}), got {logits.shape}"

    @pytest.mark.parametrize("patch_size,expected_tokens", PATCH_CONFIGS)
    def test_token_count(self, patch_size, expected_tokens):
        """Verify each patch config produces the expected number of tokens."""
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
            patch_size=patch_size,
        )
        # num_patches = grid_h * grid_w (excluding CLS token)
        num_patches = model.backbone.patch_embed.num_patches
        assert (
            num_patches == expected_tokens
        ), f"patch_size={patch_size}: expected {expected_tokens} tokens, got {num_patches}"

    @pytest.mark.parametrize("patch_size,expected_tokens", PATCH_CONFIGS)
    def test_forward_all_configs(self, batch_waveform, patch_size, expected_tokens):
        """All 4 patch configs should produce logits of shape (batch, num_classes)."""
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
            patch_size=patch_size,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch_waveform)
        assert logits.shape == (
            4,
            NUM_CLASSES,
        ), f"patch_size={patch_size}: expected (4, {NUM_CLASSES}), got {logits.shape}"

    def test_no_nans_in_output(self, batch_waveform):
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch_waveform)
        assert not torch.isnan(logits).any(), "Logits contain NaN"

    def test_no_infs_in_output(self, batch_waveform):
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batch_waveform)
        assert not torch.isinf(logits).any(), "Logits contain Inf"

    def test_mel_spec_is_nnaudio(self):
        """Model should use NNAudioMelSpectrogram (nn.Module, not torchaudio)."""
        from birdset_benchmark.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
        )
        assert isinstance(model.mel_spec, NNAudioMelSpectrogram)

    def test_single_channel_input(self):
        """Patch embed should accept 1-channel input (not 3)."""
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
        )
        conv = model.backbone.patch_embed.proj
        assert conv.in_channels == 1, f"Expected 1 input channel, got {conv.in_channels}"

    @pytest.mark.parametrize("patch_size,expected_tokens", PATCH_CONFIGS)
    def test_pos_embed_shape(self, patch_size, expected_tokens):
        """Positional embeddings should be (1, num_tokens + 1, embed_dim).

        The +1 accounts for the CLS token. embed_dim is 384 for vit_small.
        """
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
            patch_size=patch_size,
        )
        pos_embed = model.backbone.pos_embed
        expected_shape = (1, expected_tokens + 1, 384)  # +1 for CLS, 384 = vit_small embed_dim
        assert (
            pos_embed.shape == expected_shape
        ), f"patch_size={patch_size}: expected pos_embed {expected_shape}, got {pos_embed.shape}"


# ---------------------------------------------------------------------------
# GPU integration tests
# ---------------------------------------------------------------------------


class TestGPUIntegration:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_model_forward_on_gpu(self, batch_waveform):
        """Full pipeline on GPU: waveforms -> mel (GPU) -> ViT -> logits."""
        device = torch.device("cuda:0")
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
        ).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(batch_waveform.to(device))
        assert logits.device.type == "cuda", f"Expected cuda, got {logits.device}"
        assert logits.shape == (4, NUM_CLASSES)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_mel_computed_on_gpu(self, batch_waveform):
        """Mel spectrogram should be computed on GPU, not CPU."""
        device = torch.device("cuda:0")
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
        ).to(device)
        model.eval()

        # Hook into mel_spec output to verify device
        mel_device = {}

        def hook(module, input, output):
            mel_device["device"] = output.device
            mel_device["shape"] = output.shape

        model.mel_spec.register_forward_hook(hook)

        with torch.no_grad():
            model(batch_waveform.to(device))

        assert (
            mel_device["device"].type == "cuda"
        ), f"Mel was computed on {mel_device['device']}, expected cuda"
        assert mel_device["shape"] == (4, 1, 128, 500)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    @pytest.mark.parametrize("patch_size,expected_tokens", PATCH_CONFIGS)
    def test_all_configs_on_gpu(self, batch_waveform, patch_size, expected_tokens):
        """All patch configs should work end-to-end on GPU."""
        device = torch.device("cuda:0")
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
            patch_size=patch_size,
        ).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(batch_waveform.to(device))
        assert logits.device.type == "cuda"
        assert logits.shape == (4, NUM_CLASSES)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_waveform_dataset_to_gpu_pipeline(self, waveform_dataset):
        """Simulate training loop: DataLoader -> GPU model."""
        from torch.utils.data import DataLoader

        device = torch.device("cuda:0")
        model = FrameTokenizedViT(
            model_name="vit_small_patch16_224",
            num_classes=NUM_CLASSES,
        ).to(device)
        model.eval()

        loader = DataLoader(waveform_dataset, batch_size=4, shuffle=False)
        batch_audio, batch_labels = next(iter(loader))

        with torch.no_grad():
            logits = model(batch_audio.to(device))

        assert logits.device.type == "cuda"
        assert logits.shape == (4, NUM_CLASSES)
        assert batch_labels.shape == (4, NUM_CLASSES)
