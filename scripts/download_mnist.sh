#!/usr/bin/env bash
# Descarca MNIST (format IDX) in data/. Idempotent; incearca mai multe mirror-uri.
set -euo pipefail
DIR="${1:-$(dirname "$0")/../data}"
mkdir -p "$DIR"; cd "$DIR"
FILES=(train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte)
MIRRORS=(
  "https://storage.googleapis.com/cvdf-datasets/mnist"
  "https://ossci-datasets.s3.amazonaws.com/mnist"
)
fetch() {  # url out
  if command -v wget >/dev/null; then wget -q -O "$2" "$1"; else curl -fsSL -o "$2" "$1"; fi
}
for f in "${FILES[@]}"; do
  [ -s "$f" ] && continue
  ok=0
  for m in "${MIRRORS[@]}"; do
    if fetch "$m/$f.gz" "$f.gz"; then gunzip -f "$f.gz"; ok=1; break; fi
    rm -f "$f.gz"
  done
  [ $ok -eq 1 ] || { echo "EROARE: nu pot descarca $f" >&2; exit 1; }
  echo "ok: $f"
done
echo "MNIST gata in $(pwd)"
