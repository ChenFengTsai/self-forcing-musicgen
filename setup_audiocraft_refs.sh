#!/usr/bin/env bash
set -euo pipefail

BASE="/storage/ssd1/richtsai1103/MusicBench"
REF="$BASE/reference"

echo "Creating directories..."
mkdir -p "$REF/fad" "$REF/clap" "$REF/tools"

echo "Downloading VGGish checkpoint..."
wget -O "$REF/fad/vggish_model.ckpt" \
  "https://storage.googleapis.com/audioset/vggish_model.ckpt"

echo "Downloading VGGish PCA params..."
wget -O "$REF/fad/vggish_pca_params.npz" \
  "https://storage.googleapis.com/audioset/vggish_pca_params.npz"

echo "Downloading CLAP checkpoint..."
wget -O "$REF/clap/music_audioset_epoch_15_esc_90.14.pt" \
  "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"

echo "Cloning google-research for FAD code..."
if [ ! -d "$REF/tools/google-research/.git" ]; then
  git clone https://github.com/google-research/google-research.git "$REF/tools/google-research"
else
  echo "google-research already exists, skipping clone."
fi

echo
echo "Done. Verifying files:"
ls -lh \
  "$REF/fad/vggish_model.ckpt" \
  "$REF/fad/vggish_pca_params.npz" \
  "$REF/clap/music_audioset_epoch_15_esc_90.14.pt"

echo
echo "Use these env vars when running:"
echo "export AUDIOCRAFT_REFERENCE_DIR=$REF"
echo "export AUDIOCRAFT_DORA_DIR=$BASE/dora_outputs"

echo
echo "Then run Dora with this extra override for FAD:"
echo "metrics.fad.tf.bin=$REF/tools/google-research"
