#!/usr/bin/env bash
# Re-encode BirdCLEF train_soundscapes OGG -> MP3 for Raven Pro inspection.
#
# Source is Vorbis ~64-86 kbps mono @ 32 kHz. Lossy->lossy compounds artifacts,
# so target ~2x source bitrate. Default: CBR 128 kbps mono @ 32 kHz.
# (LAME VBR -V scales down for mono+32kHz; -V 5 here lands ~60 kbps, *below* source,
# which defeats the purpose. CBR is predictable. Use -q for VBR if desired.)
#
# Defaults to one labeled soundscape per site (9 files) so a single run gives a
# diverse Raven Pro starter set. Idempotent: skips files whose MP3 already exists.
#
# Usage:
#   ./encode_mp3.sh                          # 9 labeled soundscapes, one per site
#   ./encode_mp3.sh -n 20                    # first 20 labeled soundscapes
#   ./encode_mp3.sh -n all                   # all 66 labeled soundscapes
#   ./encode_mp3.sh -b 192k                  # higher CBR
#   ./encode_mp3.sh -q 2                     # VBR (~80-100 kbps mono at 32 kHz)
#   ./encode_mp3.sh -o /custom/out/dir       # custom output dir
#   ./encode_mp3.sh -s train_audio -n 50     # encode focal recordings instead

set -euo pipefail

SRC_ROOT="/mnt/data/scratch/birdclef-2026"
LABEL_CSV="${SRC_ROOT}/train_soundscapes_labels.csv"
OUT_DIR="${SRC_ROOT}/raven_pro_mp3"
SOURCE="train_soundscapes"   # or train_audio
N_FILES="per_site"           # per_site | <int> | all
CBR_BITRATE="128k"           # default CBR (predictable; ~2x source)
VBR_Q=""                     # if set, overrides CBR (LAME -q:a; 0=best, 9=worst)

usage() { sed -n '2,17p' "$0"; exit 1; }

while getopts ":o:s:n:q:b:h" opt; do
  case "$opt" in
    o) OUT_DIR="$OPTARG" ;;
    s) SOURCE="$OPTARG" ;;
    n) N_FILES="$OPTARG" ;;
    q) VBR_Q="$OPTARG"; CBR_BITRATE="" ;;
    b) CBR_BITRATE="$OPTARG"; VBR_Q="" ;;
    h|*) usage ;;
  esac
done

SRC_DIR="${SRC_ROOT}/${SOURCE}"
[[ -d "$SRC_DIR" ]] || { echo "ERROR: source dir not found: $SRC_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR"

# Build file list.
declare -a FILES
if [[ "$SOURCE" == "train_soundscapes" && "$N_FILES" == "per_site" ]]; then
  # One labeled soundscape per site.
  while IFS= read -r f; do FILES+=("$SRC_DIR/$f"); done < <(
    awk -F',' 'NR>1 {print $1}' "$LABEL_CSV" \
      | sort -u \
      | awk -F'_' '{site=$4; if (!(site in seen)) {seen[site]=1; print $0}}'
  )
elif [[ "$SOURCE" == "train_soundscapes" && "$N_FILES" == "all" ]]; then
  while IFS= read -r f; do FILES+=("$SRC_DIR/$f"); done < <(
    awk -F',' 'NR>1 {print $1}' "$LABEL_CSV" | sort -u
  )
elif [[ "$SOURCE" == "train_soundscapes" ]]; then
  while IFS= read -r f; do FILES+=("$SRC_DIR/$f"); done < <(
    awk -F',' 'NR>1 {print $1}' "$LABEL_CSV" | sort -u | head -n "$N_FILES"
  )
else
  # train_audio: just take first N OGGs in lex order.
  [[ "$N_FILES" == "per_site" || "$N_FILES" == "all" ]] && N_FILES=50
  while IFS= read -r f; do FILES+=("$f"); done < <(
    find "$SRC_DIR" -name '*.ogg' | sort | head -n "$N_FILES"
  )
fi

echo "Source dir: $SRC_DIR"
echo "Output dir: $OUT_DIR"
echo "Files:      ${#FILES[@]}"
if [[ -n "$CBR_BITRATE" ]]; then
  echo "Encoding:   CBR $CBR_BITRATE mono @ 32 kHz"
else
  echo "Encoding:   VBR -q:a $VBR_Q mono @ 32 kHz (LAME)"
fi
echo

# ffmpeg may not be on PATH (NixOS); pull it in via nix shell if needed.
if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG=(ffmpeg)
else
  echo "ffmpeg not on PATH; using 'nix shell nixpkgs#ffmpeg -c ffmpeg'"
  FFMPEG=(nix shell nixpkgs#ffmpeg -c ffmpeg)
fi

skipped=0
encoded=0
for src in "${FILES[@]}"; do
  base="$(basename "$src" .ogg)"
  # Preserve relative subdirs for train_audio (which is taxa/<species>/<file>.ogg).
  if [[ "$SOURCE" == "train_audio" ]]; then
    rel="${src#$SRC_DIR/}"
    out="$OUT_DIR/${rel%.ogg}.mp3"
    mkdir -p "$(dirname "$out")"
  else
    out="$OUT_DIR/${base}.mp3"
  fi

  if [[ -s "$out" ]]; then
    skipped=$((skipped+1))
    continue
  fi

  if [[ -n "$CBR_BITRATE" ]]; then
    "${FFMPEG[@]}" -hide_banner -loglevel error -nostdin \
      -i "$src" -ac 1 -ar 32000 -c:a libmp3lame -b:a "$CBR_BITRATE" \
      "$out"
  else
    "${FFMPEG[@]}" -hide_banner -loglevel error -nostdin \
      -i "$src" -ac 1 -ar 32000 -c:a libmp3lame -q:a "$VBR_Q" \
      "$out"
  fi
  encoded=$((encoded+1))
  echo "  -> $out"
done

echo
echo "Done. Encoded: $encoded, skipped (already existed): $skipped"
