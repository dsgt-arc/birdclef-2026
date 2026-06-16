"""Exp 038 — stream-extract core tests (basename match across prefixes, <code>/<file> layout, resumable)."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
from pathlib import Path

SCRIPTS = Path(__file__).parent.parent.parent / "experiments/038-xc-pretraining/scripts"
spec = importlib.util.spec_from_file_location("dl_birdset", SCRIPTS / "download_birdset_audio.py")
dl = importlib.util.module_from_spec(spec)
sys.modules["dl_birdset"] = dl
spec.loader.exec_module(dl)


def _make_tar(path: Path, members: dict[str, bytes]):
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_extract_basename_match_and_layout(tmp_path):
    tar = tmp_path / "shard.tar.gz"
    _make_tar(
        tar,
        {
            "XC1.ogg": b"a",  # bare → wanted
            "data/xeno-canto/update_1/ar/codeB/XC2.ogg": b"bb",  # prefixed → wanted (basename match)
            "XC3.ogg": b"ccc",  # not wanted
        },
    )
    out = tmp_path / "audio"
    got = dl.extract_wanted_from_tar(tar, {"XC1.ogg": "codeA", "XC2.ogg": "codeB"}, out)
    assert set(got) == {"XC1.ogg", "XC2.ogg"}
    assert (out / "codeA" / "XC1.ogg").read_bytes() == b"a"
    assert (out / "codeB" / "XC2.ogg").read_bytes() == b"bb"  # prefixed member → flat by code
    assert not (out / "codeA" / "XC3.ogg").exists()
    assert not any(p.name == "XC3.ogg" for p in out.rglob("*"))  # unwanted never extracted
    assert not any(p.suffix == ".tmp" for p in out.rglob("*"))  # no leftover temp files


def test_resumable_skips_existing(tmp_path):
    tar = tmp_path / "shard.tar.gz"
    _make_tar(tar, {"XC1.ogg": b"a", "XC2.ogg": b"bb"})
    out = tmp_path / "audio"
    wanted = {"XC1.ogg": "codeA", "XC2.ogg": "codeA"}
    assert set(dl.extract_wanted_from_tar(tar, wanted, out)) == {"XC1.ogg", "XC2.ogg"}
    # second pass: dests exist → nothing re-extracted
    assert dl.extract_wanted_from_tar(tar, wanted, out) == []
