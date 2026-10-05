#!/usr/bin/env bash
# Bounded software smoke: two partial epochs, ODE/SDE grids and posterior diagnostics.
# This is not a paper training configuration or a pretrained model.
set -euo pipefail
root="${1:-./mnist-smoke}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
pfm-prepare-data --dataset mnist --data-root "$root/data" --download
train_args=(--config configs/mnist_unet.yaml --variant pfm_delta --task-id smoke
  --data-root "$root/data" --output-dir "$root/runs" --device cpu
  --set checkpoint_every_epochs=1 --set batch_size=4 --set mixed_precision=none
  --set unet_base_channels=8 --set 'unet_channel_mult=[1,2]'
  --set unet_norm_groups=4 --set unet_num_res_blocks=1 --set limit_train_batches=2 --quiet)
pfm-train "${train_args[@]}" --set epochs=1
variant="$root/runs/experiments/smoke/pfm_delta"
pfm-train "${train_args[@]}" --set epochs=2 --resume "$variant/recovery/latest.json"
checkpoint="$variant/epochs/epoch_0002/checkpoint.pt"
for solver in ode sde; do
  diffusion_scale=0
  if [[ "$solver" == sde ]]; then diffusion_scale=0.1; fi
  pfm-sample --checkpoint "$checkpoint" --config configs/sampling/mnist_visualization.yaml \
    --output-dir "$root/$solver" --device cpu --set class_count=2 --set num_per_class=2 \
    --set batch_size=2 --set sampling.steps=2 --set sampling.precision=fp32 \
    --set "sampling.solver=$solver" --set "sampling.diffusion_scale=$diffusion_scale" \
    --set sampling.diffusion_schedule=linear_decay
done
pfm-evaluate --checkpoint "$checkpoint" --config configs/evaluation/mnist_posterior.yaml \
  --data-root "$root/data" --output-dir "$root/posterior" --device cpu \
  --set posterior_population=8 --set batch_size=4 --set num_workers=0 \
  --set num_bins=2 --set precision=fp32 --set 'posterior_t_values=[0.25,0.5]'
