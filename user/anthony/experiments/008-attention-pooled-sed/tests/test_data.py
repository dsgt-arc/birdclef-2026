"""Tests for data pipeline: token label construction, collation, chunking."""

import torch

from src.data import (
    _build_token_labels,
    _collate_fn,
    _parse_events,
    BIRDSET_NUM_CLASSES,
)


class TestParseEvents:
    def test_dict_format(self):
        events = [{"start_time": 1.0, "end_time": 2.0}, {"start_time": 5.0, "end_time": 6.5}]
        parsed = _parse_events(events)
        assert len(parsed) == 2
        assert parsed[0] == (1.0, 2.0)
        assert parsed[1] == (5.0, 6.5)

    def test_tuple_format(self):
        events = [(0.5, 1.5), (3.0, 4.0)]
        parsed = _parse_events(events)
        assert len(parsed) == 2

    def test_empty(self):
        assert _parse_events(None) == []
        assert _parse_events([]) == []

    def test_alt_dict_keys(self):
        events = [{"start": 1.0, "end": 2.0}]
        parsed = _parse_events(events)
        assert parsed[0] == (1.0, 2.0)


class TestBuildTokenLabels:
    def test_no_events_all_zeros(self):
        labels = _build_token_labels(
            events=[], chunk_start_s=0.0,
            num_tokens=100, hop_s=0.01,
            num_classes=21, label_vec=torch.ones(21),
        )
        assert labels.shape == (100, 21)
        assert labels.sum() == 0

    def test_single_event_covers_correct_tokens(self):
        """Event from 0.5s to 0.6s should activate tokens 50-59 (0-indexed)."""
        label_vec = torch.zeros(21)
        label_vec[0] = 1.0

        labels = _build_token_labels(
            events=[(0.5, 0.6)],
            chunk_start_s=0.0,
            num_tokens=1000,
            hop_s=0.01,  # 10ms per token
            num_classes=21,
            label_vec=label_vec,
        )

        # Tokens 50-59 should be active
        for i in range(1000):
            if 50 <= i < 60:
                assert labels[i, 0] == 1.0, f"Token {i} should be active"
            else:
                assert labels[i, 0] == 0.0, f"Token {i} should be inactive"

    def test_event_at_chunk_boundary(self):
        """Event starting before chunk: only the overlapping part should be labeled."""
        label_vec = torch.ones(5)

        # Chunk starts at 5.0s, event spans 4.9s-5.2s
        labels = _build_token_labels(
            events=[(4.9, 5.2)],
            chunk_start_s=5.0,
            num_tokens=100,
            hop_s=0.1,
            num_classes=5,
            label_vec=label_vec,
        )

        # Tokens 0-1 (5.0-5.1, 5.1-5.2) should be active
        assert labels[0].sum() > 0
        assert labels[1].sum() > 0
        assert labels[2].sum() == 0

    def test_event_outside_chunk(self):
        """Event entirely outside chunk should produce all-zero labels."""
        label_vec = torch.ones(5)

        labels = _build_token_labels(
            events=[(0.0, 1.0)],
            chunk_start_s=5.0,
            num_tokens=100,
            hop_s=0.1,  # chunk: 5.0-15.0s
            num_classes=5,
            label_vec=label_vec,
        )

        assert labels.sum() == 0

    def test_multiple_events(self):
        """Multiple events should independently activate tokens."""
        label_vec = torch.zeros(3)
        label_vec[0] = 1.0

        labels = _build_token_labels(
            events=[(0.0, 0.1), (0.5, 0.6)],
            chunk_start_s=0.0,
            num_tokens=100,
            hop_s=0.01,
            num_classes=3,
            label_vec=label_vec,
        )

        # Both regions should be active
        assert labels[0:10, 0].sum() == 10  # first event
        assert labels[50:60, 0].sum() == 10  # second event
        assert labels[20:40, 0].sum() == 0  # gap between events

    def test_label_vec_propagated(self):
        """Token labels should copy the full label vector, not just class 0."""
        label_vec = torch.zeros(5)
        label_vec[2] = 1.0
        label_vec[4] = 1.0

        labels = _build_token_labels(
            events=[(0.0, 0.1)],
            chunk_start_s=0.0,
            num_tokens=100,
            hop_s=0.01,
            num_classes=5,
            label_vec=label_vec,
        )

        assert labels[5, 2] == 1.0
        assert labels[5, 4] == 1.0
        assert labels[5, 0] == 0.0


class TestCollation:
    def test_collate_shapes(self):
        """Collate should produce correct batch shapes."""
        batch = [
            {
                "waveform": torch.randn(320000),
                "clip_labels": torch.zeros(21),
                "token_labels": torch.zeros(1000, 21),
                "real_tokens": 1000,
            },
            {
                "waveform": torch.randn(320000),
                "clip_labels": torch.zeros(21),
                "token_labels": torch.zeros(1000, 21),
                "real_tokens": 500,
            },
        ]

        waveforms, clip_labels, token_labels, padding_mask = _collate_fn(batch)

        assert waveforms.shape == (2, 320000)
        assert clip_labels.shape == (2, 21)
        assert token_labels.shape == (2, 1000, 21)
        assert padding_mask.shape == (2, 1000)

    def test_padding_mask_values(self):
        """Padding mask should be True for real tokens, False for padding."""
        batch = [
            {"waveform": torch.randn(320000), "clip_labels": torch.zeros(5),
             "token_labels": torch.zeros(100, 5), "real_tokens": 100},
            {"waveform": torch.randn(320000), "clip_labels": torch.zeros(5),
             "token_labels": torch.zeros(100, 5), "real_tokens": 60},
        ]

        _, _, _, padding_mask = _collate_fn(batch)

        # First sample: all real
        assert padding_mask[0].all()
        # Second sample: 60 real, 40 padding
        assert padding_mask[1, :60].all()
        assert not padding_mask[1, 60:].any()


class TestBirdSetNumClasses:
    def test_hsn_classes(self):
        assert BIRDSET_NUM_CLASSES["HSN"] == 21
