"""WavTokenizer encode/decode for focal recordings and soundscape windows.

Loads pretrained WavTokenizer, encodes audio to discrete tokens, decodes
back to wav for round-trip quality evaluation. Also extracts codebook
embeddings for super-token quantization.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import typer
import yaml
from huggingface_hub import hf_hub_download

app = typer.Typer()


def _ensure_wavtokenizer_importable(repo_root: Path):
    """Add vendor/WavTokenizer to sys.path for imports."""
    wavtok_dir = repo_root / "vendor" / "WavTokenizer"
    if not wavtok_dir.exists():
        raise FileNotFoundError(
            f"WavTokenizer vendor dir not found: {wavtok_dir}\n"
            "Run: git submodule update --init vendor/WavTokenizer"
        )
    if str(wavtok_dir) not in sys.path:
        sys.path.insert(0, str(wavtok_dir))


def _download_model(repo_id: str, config_filename: str = None, model_filename: str = None):
    """Download WavTokenizer config and checkpoint from HuggingFace.

    If filenames are not specified, lists repo files and picks the .yaml/.ckpt.
    Returns paths in the HF cache (no copy needed).
    """
    from huggingface_hub import list_repo_files

    print(f"Ensuring model files from {repo_id}...")

    if config_filename is None or model_filename is None:
        repo_files = list(list_repo_files(repo_id))
        print(f"  Repo files: {repo_files}")
        if config_filename is None:
            yaml_files = [f for f in repo_files if f.endswith(".yaml")]
            if not yaml_files:
                raise FileNotFoundError(f"No .yaml config in {repo_id}: {repo_files}")
            config_filename = yaml_files[0]
        if model_filename is None:
            ckpt_files = [f for f in repo_files if f.endswith(".ckpt")]
            if not ckpt_files:
                raise FileNotFoundError(f"No .ckpt model in {repo_id}: {repo_files}")
            model_filename = ckpt_files[0]

    print(f"  Config: {config_filename}")
    print(f"  Model: {model_filename}")
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
    return config_path, model_path


def _load_audio(path: str, target_sr: int, max_duration_sec: float = 60.0) -> torch.Tensor:
    """Load audio, convert to mono, resample to target_sr, cap duration."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = torch.from_numpy(audio).float()
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
    max_samples = int(max_duration_sec * target_sr)
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    return audio


def _extract_codebook(model) -> np.ndarray:
    """Extract codebook embeddings from WavTokenizer feature extractor."""
    vq_layers = model.feature_extractor.encodec.quantizer.vq.layers
    # Single codebook (nq=1), take first layer
    codebook = vq_layers[0].codebook.detach().cpu().numpy()
    return codebook


@app.command()
def main(
    config: str = "configs/exp104.yaml",
    skip_soundscapes: bool = False,
):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Resolve repo root (go up from experiment dir)
    # __file__ = .../104-wavtokenizer-retrieval/src/birdclef_104/tokenize.py
    # parents[2] = .../104-wavtokenizer-retrieval/
    # parents[3] = .../experiments/  parents[4] = .../anthony/  parents[5] = .../user/
    exp_dir = Path(__file__).resolve().parents[2]
    repo_root = exp_dir.parents[3]  # user/anthony/experiments/NNN -> birdclef-2026
    _ensure_wavtokenizer_importable(repo_root)

    from decoder.pretrained import WavTokenizer

    focal_dir = Path(cfg["data"]["focal_dir"])
    soundscape_dir = Path(cfg["data"]["soundscape_dir"])
    target_sr = cfg["wavtokenizer"]["sample_rate"]
    max_duration = cfg["subset"].get("max_duration_sec", 60.0)
    output_root = Path(cfg["output"]["root"])

    # Download / locate model
    config_path, model_path = _download_model(
        repo_id=cfg["wavtokenizer"]["repo_id"],
        config_filename=cfg["wavtokenizer"].get("config_filename"),
        model_filename=cfg["wavtokenizer"].get("model_filename"),
    )

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading WavTokenizer on {device}...")
    model = WavTokenizer.from_pretrained0802(config_path, model_path)
    model = model.to(device)
    model.eval()

    # Extract codebook embeddings
    codebook = _extract_codebook(model)
    codebook_path = output_root / "codebook_embeddings.npy"
    np.save(codebook_path, codebook)
    print(f"Codebook: {codebook.shape} saved to {codebook_path}")

    # Load subset
    subset_path = output_root / "subset.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"Run select_subset first: {subset_path}")
    with open(subset_path) as f:
        subset = json.load(f)

    # Encode/decode focal recordings
    tokens_dir = output_root / "tokens"
    recon_dir = output_root / "reconstructed"
    bandwidth_id = torch.tensor([0], device=device)

    total_files = sum(len(info["files"]) for info in subset.values())
    processed = 0

    for sp_id, info in subset.items():
        sp_tokens_dir = tokens_dir / sp_id
        sp_recon_dir = recon_dir / sp_id
        sp_tokens_dir.mkdir(parents=True, exist_ok=True)
        sp_recon_dir.mkdir(parents=True, exist_ok=True)

        for filename in info["files"]:
            audio_path = focal_dir / filename
            stem = Path(filename).stem

            if not audio_path.exists():
                print(f"  SKIP (not found): {audio_path}")
                continue

            try:
                audio = _load_audio(str(audio_path), target_sr, max_duration)
                wav = audio.unsqueeze(0).to(device)  # (1, T)

                # Encode
                features, discrete_codes = model.encode_infer(wav, bandwidth_id=bandwidth_id)
                # Shape: (nq=1, B=1, L) -> (L,); squeeze only known dims
                tokens = discrete_codes[0, 0].cpu().numpy().astype(np.int32)
                np.save(sp_tokens_dir / f"{stem}.npy", tokens)

                # Decode
                audio_out = model.decode(features, bandwidth_id=bandwidth_id)
                audio_out = audio_out.squeeze().cpu().numpy()
                sf.write(str(sp_recon_dir / f"{stem}.wav"), audio_out, target_sr)

                processed += 1
                if processed % 10 == 0:
                    print(f"  [{processed}/{total_files}] {filename} -> {len(tokens)} tokens "
                          f"({len(tokens)/target_sr*320:.1f}s)")
            except Exception as e:
                print(f"  ERROR {filename}: {e}")

    print(f"\nDone: {processed}/{total_files} focal recordings tokenized")
    print(f"Tokens: {tokens_dir}")
    print(f"Reconstructed: {recon_dir}")

    # Optionally tokenize soundscape windows for Phase 3
    if not skip_soundscapes:
        _tokenize_soundscapes(cfg, model, device, bandwidth_id, target_sr, output_root)


def _tokenize_soundscapes(cfg, model, device, bandwidth_id, target_sr, output_root):
    """Tokenize labeled soundscape windows (5s each) for Phase 3."""
    import pandas as pd

    soundscape_dir = Path(cfg["data"]["soundscape_dir"])
    labels_csv = cfg["data"]["soundscape_labels_csv"]
    labels_df = pd.read_csv(labels_csv)

    sc_tokens_dir = output_root / "soundscape_tokens"
    sc_tokens_dir.mkdir(parents=True, exist_ok=True)

    # Get unique filenames
    filenames = labels_df["filename"].unique()
    print(f"\nTokenizing {len(filenames)} soundscape files ({len(labels_df)} windows)...")

    for i, fname in enumerate(filenames):
        audio_path = soundscape_dir / fname
        if not audio_path.exists():
            print(f"  SKIP: {audio_path}")
            continue

        try:
            # Load full file
            audio = _load_audio(str(audio_path), target_sr, max_duration_sec=120.0)
            wav = audio.unsqueeze(0).to(device)

            # Encode full file
            features, discrete_codes = model.encode_infer(wav, bandwidth_id=bandwidth_id)
            tokens = discrete_codes.squeeze().cpu().numpy().astype(np.int32)

            stem = Path(fname).stem
            np.save(sc_tokens_dir / f"{stem}.npy", tokens)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(filenames)}] {fname} -> {len(tokens)} tokens")
        except Exception as e:
            print(f"  ERROR {fname}: {e}")

    print(f"Soundscape tokens: {sc_tokens_dir}")


if __name__ == "__main__":
    app()
