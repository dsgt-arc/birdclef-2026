"""Tests for teacher_labels.utils — no data dependencies."""

from pathlib import Path

from teacher_labels.utils import partition_files


def test_partition_files_modular():
    """Partition uses modular arithmetic: file i goes to partition i % N."""
    files = [Path(f"file_{i}.ogg") for i in range(10)]
    p0 = partition_files(files, 3, 0)
    p1 = partition_files(files, 3, 1)
    p2 = partition_files(files, 3, 2)

    assert [f.name for f in p0] == ["file_0.ogg", "file_3.ogg", "file_6.ogg", "file_9.ogg"]
    assert [f.name for f in p1] == ["file_1.ogg", "file_4.ogg", "file_7.ogg"]
    assert [f.name for f in p2] == ["file_2.ogg", "file_5.ogg", "file_8.ogg"]


def test_partition_files_covers_all():
    """All files appear in exactly one partition."""
    files = [Path(f"f{i}") for i in range(100)]
    all_partitioned = []
    for pid in range(7):
        all_partitioned.extend(partition_files(files, 7, pid))
    assert sorted(all_partitioned) == sorted(files)


def test_partition_files_empty():
    """Empty file list returns empty partitions."""
    assert partition_files([], 5, 0) == []


def test_partition_files_single():
    """Single file goes to partition 0."""
    files = [Path("only.ogg")]
    assert partition_files(files, 10, 0) == files
    assert partition_files(files, 10, 1) == []
