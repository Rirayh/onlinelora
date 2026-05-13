#!/usr/bin/env bash
# baselines/setup_baselines.sh
# Re-clone all upstream baseline repos at pinned commit hashes documented in MANIFEST.md.
# Run from project root:  bash baselines/setup_baselines.sh
# Idempotent: skips already-present dirs.

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

clone_pinned () {
  local dir="$1" url="$2" commit="$3"
  if [ -d "$dir/.git" ]; then
    echo "[skip] $dir already present"
    return
  fi
  echo "[clone] $dir <- $url @ $commit"
  git clone "$url" "$dir"
  ( cd "$dir" && git checkout "$commit" )
}

# Tier-1 official code (Phase B1)
clone_pinned DoRA_official        https://github.com/NVlabs/DoRA.git           7e2f10abbe8efe212c8fca1d983ae1d04ef13a18
clone_pinned AdaLoRA_official     https://github.com/QingruZhang/AdaLoRA.git   d10f5ebee16c478fa2f41a44a237b38e8c9b0338
clone_pinned ReLoRA_official      https://github.com/Guitaricet/relora.git    176f37633fe02019835387258ddabcf6d91e328d
clone_pinned LoRAPrune_reference  https://github.com/aim-uofa/LoRAPrune.git   4da52721b00cd80ef9ec2071d338d60efc7024e7

echo ""
echo "[done] all official baselines cloned to baselines/"
echo "       Phase B1.5 baselines (COLA / Sensitivity-LoRA / CTR-LoRA / PrunedLoRA)"
echo "       have no public code — see *_reimpl/PAPER.pdf and re-implement per IMPLEMENTATION_NOTES.md."
