#!/bin/bash
#SBATCH --job-name=download-ckpt-002
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t60 -qembers
#SBATCH --output=logs/download-checkpoints-%j.out
set -eu

echo "Downloading BEATs checkpoints for experiment 002"
hostname
date

scratch_dir=~/scratch/birdclef-2026
beats_dir=$scratch_dir/checkpoints/beats
repo_dir=~/arc/birdclef-2026

# Create ephemeral venv in TMPDIR (fast local disk)
LOCAL_VENV=$TMPDIR/venv
uv venv $LOCAL_VENV
source $LOCAL_VENV/bin/activate

# Set HuggingFace cache to scratch (persistent)
export HF_HOME=$scratch_dir/.cache/huggingface
mkdir -p $HF_HOME

# Install dependencies for download
uv pip install huggingface_hub playwright

# Install playwright browsers (chromium only)
playwright install chromium

mkdir -p $beats_dir

# ============================================================
# BEATs Encoder (via HuggingFace mirror)
# ============================================================
echo ""
echo "=== Downloading BEATs Encoder checkpoint ==="

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os

scratch = os.path.expanduser("~/scratch/birdclef-2026")
beats_dir = f"{scratch}/checkpoints/beats"

# Download BEATs_iter3.pt (base encoder, not finetuned)
print("Downloading BEATs_iter3.pt (base encoder)...")
hf_hub_download(
    repo_id="lpepino/beats_ckpts",
    filename="BEATs_iter3.pt",
    local_dir=beats_dir,
)
print("BEATs encoder downloaded!")

# Also ensure we have the finetuned version for comparison
print("Downloading BEATs_iter3_plus_AS2M.pt (finetuned)...")
hf_hub_download(
    repo_id="lpepino/beats_ckpts",
    filename="BEATs_iter3_plus_AS2M.pt",
    local_dir=beats_dir,
)
print("BEATs finetuned model downloaded!")
EOF

# ============================================================
# BEATs Tokenizer (via Playwright for OneDrive)
# ============================================================
echo ""
echo "=== Downloading BEATs Tokenizer checkpoint ==="

TOKENIZER_PATH=$beats_dir/Tokenizers_iter3.pt
TOKENIZER_URL="https://1drv.ms/u/s!AqeByhGUtINrgcp1DEzUBtzHapxcqw?e=JZI5Uf"

if [ -f "$TOKENIZER_PATH" ]; then
    echo "Tokenizers_iter3.pt already exists, skipping download"
    ls -la $TOKENIZER_PATH
else
    echo "Downloading Tokenizers_iter3.pt from OneDrive using Playwright..."

    python3 << EOF
"""Download file from OneDrive using Playwright browser automation."""
from pathlib import Path
from playwright.sync_api import sync_playwright

share_url = "$TOKENIZER_URL"
output_path = Path("$TOKENIZER_PATH").expanduser().resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading from: {share_url}")
print(f"Saving to: {output_path}")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()

    try:
        print("Loading OneDrive page...")
        page.goto(share_url, timeout=60000)
        page.wait_for_load_state("networkidle", timeout=30000)

        # Try multiple download button selectors
        download_selectors = [
            'button[data-automationid="downloadCommand"]',
            '[aria-label="Download"]',
            'button:has-text("Download")',
            '[data-icon-name="Download"]',
        ]

        download_button = None
        for selector in download_selectors:
            try:
                download_button = page.wait_for_selector(selector, timeout=5000)
                if download_button:
                    print(f"Found download button: {selector}")
                    break
            except Exception:
                continue

        if download_button:
            with page.expect_download(timeout=300000) as download_info:
                download_button.click()
            download = download_info.value
            print(f"Download started: {download.suggested_filename}")
            download.save_as(str(output_path))

            if output_path.exists() and output_path.stat().st_size > 0:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"Download complete! Size: {size_mb:.1f} MB")
            else:
                raise Exception("Downloaded file is empty or missing")
        else:
            raise Exception("Could not find download button")

    except Exception as e:
        print(f"ERROR: {e}")
        # Save debug screenshot
        debug_path = output_path.parent / "download_debug.png"
        page.screenshot(path=str(debug_path))
        print(f"Debug screenshot: {debug_path}")
        raise
    finally:
        browser.close()
EOF

    if [ ! -f "$TOKENIZER_PATH" ]; then
        echo ""
        echo "WARNING: Automated download failed!"
        echo ""
        echo "Please download manually:"
        echo "  1. Open in browser: $TOKENIZER_URL"
        echo "  2. Click 'Download' button"
        echo "  3. Upload to cluster:"
        echo "     scp Tokenizers_iter3.pt <user>@login-hpc.pace.gatech.edu:$TOKENIZER_PATH"
        echo ""
    fi
fi

# ============================================================
# Verify Downloads
# ============================================================
echo ""
echo "=== Verifying checkpoints ==="

python3 << EOF
import torch
import sys
sys.path.insert(0, "$repo_dir/vendor/unilm/beats")

from BEATs import BEATs, BEATsConfig

beats_dir = "$beats_dir"

# Check encoder
encoder_path = f"{beats_dir}/BEATs_iter3.pt"
try:
    ckpt = torch.load(encoder_path, map_location='cpu', weights_only=False)
    cfg = BEATsConfig(ckpt['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(ckpt['model'])
    print(f"BEATs_iter3.pt: OK (embed_dim={cfg.encoder_embed_dim})")
except Exception as e:
    print(f"BEATs_iter3.pt: FAILED - {e}")

# Check tokenizer if exists
tokenizer_path = f"{beats_dir}/Tokenizers_iter3.pt"
import os
if os.path.exists(tokenizer_path):
    try:
        from Tokenizers import Tokenizers, TokenizersConfig
        ckpt = torch.load(tokenizer_path, map_location='cpu', weights_only=False)
        cfg = TokenizersConfig(ckpt['cfg'])
        model = Tokenizers(cfg)
        model.load_state_dict(ckpt['model'])
        print(f"Tokenizers_iter3.pt: OK (quant_n={cfg.quant_n})")
    except Exception as e:
        print(f"Tokenizers_iter3.pt: FAILED - {e}")
else:
    print(f"Tokenizers_iter3.pt: MISSING")
EOF

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== Checkpoint Status ==="
echo ""
echo "BEATs checkpoints:"
ls -lh $beats_dir/*.pt 2>/dev/null || echo "  (none found)"

echo ""
echo "Required for experiment 002:"
echo "  - BEATs_iter3.pt (encoder):        $([ -f $beats_dir/BEATs_iter3.pt ] && echo 'OK' || echo 'MISSING')"
echo "  - Tokenizers_iter3.pt (tokenizer): $([ -f $TOKENIZER_PATH ] && echo 'OK' || echo 'MISSING')"

echo ""
echo "Done!"
date
