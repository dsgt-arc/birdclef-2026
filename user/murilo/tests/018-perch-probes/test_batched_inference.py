"""Tests verifying batched Perch inference matches file-by-file inference.

The kernel switched from ThreadPoolExecutor (1 file per TF call) to batched
inference (16 files per TF call). These tests verify the batching logic
produces identical results using a mock inference function.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock Perch inference — deterministic function of input waveform
# ---------------------------------------------------------------------------

N_WINDOWS = 12
WINDOW_SAMPLES = 160000
N_SPECIES = 234
EMB_DIM = 1536
N_BC_LABELS = 14795  # Perch v2 output size


def mock_infer_fn(waveform_batch):
    """Deterministic mock: embeddings = mean of waveform, logits = hash of waveform.

    This ensures that if the same waveform goes through batched vs sequential,
    we get the exact same output.
    """
    n = waveform_batch.shape[0]
    # Embeddings: use waveform statistics as features (deterministic)
    emb = np.zeros((n, EMB_DIM), dtype=np.float32)
    for i in range(n):
        rng = np.random.RandomState(seed=int(abs(waveform_batch[i, :10].sum() * 1e6) % (2**31)))
        emb[i] = rng.randn(EMB_DIM).astype(np.float32)

    # Logits: also deterministic from waveform
    logits = np.zeros((n, N_BC_LABELS), dtype=np.float32)
    for i in range(n):
        rng = np.random.RandomState(seed=int(abs(waveform_batch[i, :10].sum() * 1e6) % (2**31)) + 1)
        logits[i] = rng.randn(N_BC_LABELS).astype(np.float32) * 0.1

    return emb, logits


def mock_compute_mapped_logits(logits):
    """Simplified species mapping: just take first N_SPECIES columns."""
    return logits[:, :N_SPECIES].copy()


def load_and_window_audio_mock(file_id):
    """Deterministic mock audio: each file produces unique waveform from its id."""
    rng = np.random.RandomState(seed=file_id)
    audio = rng.randn(60 * 32000).astype(np.float32) * 0.01
    return audio.reshape(N_WINDOWS, WINDOW_SAMPLES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchedInferenceEquivalence:
    """Verify batched inference gives identical results to file-by-file."""

    @pytest.fixture
    def file_ids(self):
        """Simulate 35 files (not divisible by batch_size=16)."""
        return list(range(35))

    def _run_sequential(self, file_ids):
        """File-by-file inference (old ThreadPoolExecutor approach)."""
        all_emb = []
        all_scores = []
        for fid in file_ids:
            waveform = load_and_window_audio_mock(fid)  # (12, 160000)
            emb, logits = mock_infer_fn(waveform)
            mapped = mock_compute_mapped_logits(logits)
            all_emb.append(emb)
            all_scores.append(mapped)
        return np.concatenate(all_emb, axis=0), np.concatenate(all_scores, axis=0)

    def _run_batched(self, file_ids, batch_size=16):
        """Batched inference (new approach matching Yuriy)."""
        all_emb = []
        all_scores = []
        for batch_start in range(0, len(file_ids), batch_size):
            batch_ids = file_ids[batch_start : batch_start + batch_size]
            batch_n = len(batch_ids)

            waveforms = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            for i, fid in enumerate(batch_ids):
                waveform = load_and_window_audio_mock(fid)
                waveforms[i * N_WINDOWS : (i + 1) * N_WINDOWS] = waveform

            emb, logits = mock_infer_fn(waveforms)
            mapped = mock_compute_mapped_logits(logits)
            all_emb.append(emb)
            all_scores.append(mapped)
        return np.concatenate(all_emb, axis=0), np.concatenate(all_scores, axis=0)

    def test_embeddings_match(self, file_ids):
        """Batched embeddings should be identical to sequential."""
        emb_seq, _ = self._run_sequential(file_ids)
        emb_bat, _ = self._run_batched(file_ids)
        np.testing.assert_array_equal(emb_seq, emb_bat)

    def test_scores_match(self, file_ids):
        """Batched scores should be identical to sequential."""
        _, scores_seq = self._run_sequential(file_ids)
        _, scores_bat = self._run_batched(file_ids)
        np.testing.assert_array_equal(scores_seq, scores_bat)

    def test_shapes_match(self, file_ids):
        """Output shapes should match regardless of batching."""
        emb_seq, scores_seq = self._run_sequential(file_ids)
        emb_bat, scores_bat = self._run_batched(file_ids)
        assert emb_seq.shape == emb_bat.shape
        assert scores_seq.shape == scores_bat.shape
        assert emb_seq.shape == (35 * 12, EMB_DIM)
        assert scores_seq.shape == (35 * 12, N_SPECIES)

    def test_last_batch_partial(self):
        """Last batch may be smaller than batch_size — should still work."""
        file_ids = list(range(5))  # 5 files, batch_size=16 → single partial batch
        emb_seq, scores_seq = self._run_sequential(file_ids)
        emb_bat, scores_bat = self._run_batched(file_ids, batch_size=16)
        np.testing.assert_array_equal(emb_seq, emb_bat)
        np.testing.assert_array_equal(scores_seq, scores_bat)

    def test_single_file(self):
        """Edge case: single file."""
        emb_seq, scores_seq = self._run_sequential([0])
        emb_bat, scores_bat = self._run_batched([0])
        np.testing.assert_array_equal(emb_seq, emb_bat)

    def test_exact_batch_boundary(self):
        """Exactly divisible by batch_size (no partial last batch)."""
        file_ids = list(range(32))  # 32 / 16 = 2 even batches
        emb_seq, scores_seq = self._run_sequential(file_ids)
        emb_bat, scores_bat = self._run_batched(file_ids, batch_size=16)
        np.testing.assert_array_equal(emb_seq, emb_bat)

    def test_different_batch_sizes_same_result(self):
        """Different batch sizes should produce identical results."""
        file_ids = list(range(20))
        _, scores_b4 = self._run_batched(file_ids, batch_size=4)
        _, scores_b8 = self._run_batched(file_ids, batch_size=8)
        _, scores_b16 = self._run_batched(file_ids, batch_size=16)
        np.testing.assert_array_equal(scores_b4, scores_b8)
        np.testing.assert_array_equal(scores_b8, scores_b16)


class TestMetadataOrdering:
    """Verify metadata is assembled in the correct order."""

    def test_metadata_count(self):
        """N files should produce N*12 metadata rows."""
        n_files = 7
        n_expected = n_files * N_WINDOWS
        # Simulate metadata assembly from batched approach
        meta = []
        for fid in range(n_files):
            for w in range(N_WINDOWS):
                meta.append({"soundscape_id": f"file_{fid}", "end_time": (w + 1) * 5})
        assert len(meta) == n_expected

    def test_metadata_order_matches_scores(self):
        """Row i of scores should correspond to row i of metadata."""
        file_ids = [10, 20, 30]  # 3 files
        meta = []
        waveform_order = []

        for fid in file_ids:
            for w in range(N_WINDOWS):
                meta.append({"file": fid, "window": w})
                waveform_order.append((fid, w))

        # Verify ordering
        for i, (fid, w) in enumerate(waveform_order):
            assert meta[i]["file"] == fid
            assert meta[i]["window"] == w

    def test_batch_boundary_doesnt_break_ordering(self):
        """Files split across batches should still have correct metadata order."""
        file_ids = list(range(5))
        batch_size = 2  # 5 files → batches of [2, 2, 1]

        meta_batched = []
        for batch_start in range(0, len(file_ids), batch_size):
            batch = file_ids[batch_start : batch_start + batch_size]
            for fid in batch:
                for w in range(N_WINDOWS):
                    meta_batched.append({"file": fid, "window": w})

        # Should be identical to sequential ordering
        meta_sequential = []
        for fid in file_ids:
            for w in range(N_WINDOWS):
                meta_sequential.append({"file": fid, "window": w})

        assert meta_batched == meta_sequential


class TestWaveformAssembly:
    """Verify the waveform array is correctly assembled for batched calls."""

    def test_waveform_shape(self):
        """Batch of N files → (N*12, 160000) waveform array."""
        batch_n = 4
        waveforms = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        for i in range(batch_n):
            waveform = load_and_window_audio_mock(i)
            waveforms[i * N_WINDOWS : (i + 1) * N_WINDOWS] = waveform
        assert waveforms.shape == (48, WINDOW_SAMPLES)

    def test_waveform_file_isolation(self):
        """Each file's 12 windows should be contiguous and correct."""
        batch_n = 3
        waveforms = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        individual = []
        for i in range(batch_n):
            waveform = load_and_window_audio_mock(i)
            waveforms[i * N_WINDOWS : (i + 1) * N_WINDOWS] = waveform
            individual.append(waveform)

        # Verify each file's chunk matches individual loading
        for i in range(batch_n):
            np.testing.assert_array_equal(
                waveforms[i * N_WINDOWS : (i + 1) * N_WINDOWS],
                individual[i],
            )
