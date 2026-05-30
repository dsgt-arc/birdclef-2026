"""Exp 036 Part 4 — post-hoc SWA averaging (swa_average.py).

Float tensors averaged; int buffers taken from the last ckpt; output is an
export-consumable Lightning-style {"state_dict": model.*} ckpt.
"""

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(
    "/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026"
    "/user/murilo/experiments/036-xc-backbone-finetune/scripts"
)
sys.path.insert(0, str(SCRIPTS))
import swa_average  # noqa: E402


def test_average_known_floats():
    sds = [
        {"model.w": torch.tensor([0.0, 2.0])},
        {"model.w": torch.tensor([2.0, 4.0])},
        {"model.w": torch.tensor([4.0, 6.0])},
    ]
    avg = swa_average.average_state_dicts(sds)
    torch.testing.assert_close(avg["model.w"], torch.tensor([2.0, 4.0]))  # column means


def test_int_buffer_taken_from_last_not_averaged():
    sds = [
        {"model.bn.num_batches_tracked": torch.tensor(10)},
        {"model.bn.num_batches_tracked": torch.tensor(20)},
        {"model.bn.num_batches_tracked": torch.tensor(30)},
    ]
    avg = swa_average.average_state_dicts(sds)
    assert avg["model.bn.num_batches_tracked"].item() == 30  # last, not mean (20)


def test_dtype_preserved():
    sds = [
        {"model.w": torch.tensor([1.0, 3.0], dtype=torch.float32)},
        {"model.w": torch.tensor([3.0, 5.0], dtype=torch.float32)},
    ]
    avg = swa_average.average_state_dicts(sds)
    assert avg["model.w"].dtype == torch.float32
    torch.testing.assert_close(avg["model.w"], torch.tensor([2.0, 4.0]))


def test_mismatched_keys_raise():
    with pytest.raises(ValueError):
        swa_average.average_state_dicts([{"model.a": torch.zeros(1)}, {"model.b": torch.zeros(1)}])


def test_select_ckpts_last_k_excludes_last(tmp_path):
    for e in range(5):
        torch.save(
            {"state_dict": {"model.w": torch.tensor([float(e)])}},
            tmp_path / f"epochepoch={e:02d}.ckpt",
        )
    torch.save({"state_dict": {"model.w": torch.tensor([99.0])}}, tmp_path / "last.ckpt")
    sel = swa_average.select_ckpts(tmp_path, last_k=3)
    assert [p.name for p in sel] == [
        "epochepoch=02.ckpt",
        "epochepoch=03.ckpt",
        "epochepoch=04.ckpt",
    ]  # last 3 epochs, last.ckpt excluded


def test_model_state_filters_non_model_keys(tmp_path):
    torch.save(
        {"state_dict": {"model.w": torch.tensor([1.0]), "val_auroc.x": torch.tensor([9.0])}},
        tmp_path / "epochepoch=00.ckpt",
    )
    ms = swa_average._model_state(tmp_path / "epochepoch=00.ckpt")
    assert set(ms) == {"model.w"}  # non-model.* (metric state) filtered out


def test_end_to_end_export_consumable(tmp_path):
    for e in range(3):
        torch.save(
            {"state_dict": {"model.w": torch.tensor([float(e), float(e) + 1])}},
            tmp_path / f"epochepoch={e:02d}.ckpt",
        )
    ckpts = swa_average.select_ckpts(tmp_path, last_k=3)
    avg = swa_average.average_state_dicts([swa_average._model_state(c) for c in ckpts])
    out = tmp_path / "swa.ckpt"
    torch.save({"state_dict": avg}, out)
    loaded = torch.load(out, weights_only=False)
    # export_xcpretrain.py reads state_dict + filters model.* → all keys must be model.*
    assert all(k.startswith("model.") for k in loaded["state_dict"])
    torch.testing.assert_close(loaded["state_dict"]["model.w"], torch.tensor([1.0, 2.0]))
