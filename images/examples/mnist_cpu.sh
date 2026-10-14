#!/usr/bin/env bash
set -euo pipefail
# Run from images/ with this task's environment active.
out=${1:?Usage: bash examples/mnist_cpu.sh /path/to/output [existing-MNIST-directory]}
if [[ $# -ge 2 ]]; then
  data=$2
else
  data="$out/data"
  pfm-images-prepare data --dataset mnist --data-root "$data"
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
pfm-images-train --config configs/mnist_cpu.yaml --data-root "$data" --output "$out/train" --device cpu
pfm-images-train --resume "$out/train/latest.pt" --data-root "$data" --output "$out/train" --device cpu --set epochs=2
pfm-images-sample --checkpoint "$out/train/latest.pt" --sampling configs/ode.yaml --output "$out/ode" --count 4 --set steps=2
pfm-images-sample --checkpoint "$out/train/latest.pt" --sampling configs/sde.yaml --output "$out/sde" --count 4 --set steps=2
pfm-images-evaluate posterior --checkpoint "$out/train/latest.pt" --data-root "$data" --output "$out/posterior.json" --count 8 --times 0.25 0.5 0.75
