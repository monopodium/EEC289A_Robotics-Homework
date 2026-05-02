#!/usr/bin/env bash
# Run every evaluation step after training finishes.
# Generates:
#   - artifacts/public_eval_baseline/   <- stage_1 ckpt, public benchmark
#   - artifacts/public_eval_bundle/     <- stage_2 ckpt, public benchmark
#   - artifacts/per_direction_baseline/ <- stage_1 ckpt, per-direction
#   - artifacts/per_direction_eval/     <- stage_2 ckpt, per-direction
#   - artifacts/demo_bundle/demo.mp4    <- stage_2 ckpt, qualitative video
#   - artifacts/figures/*.png           <- report figures
set -euo pipefail
cd "$(dirname "$0")"

source /home/uccl/nfs/shuangma/Homework1/env.sh

CFG=configs/colab_runtime_config.json
STAGE1_CKPT=artifacts/run_baseline/best_checkpoint_stage1
STAGE2_CKPT=artifacts/run_baseline/best_checkpoint

if [ ! -d "$STAGE1_CKPT" ]; then
  STAGE1_CKPT=$(ls -d artifacts/run_baseline/stage_1/checkpoints/* | sort | tail -1)
  echo "[run_all_evals] stage_1 best_checkpoint snapshot missing, using latest: $STAGE1_CKPT"
fi

echo "==> public benchmark on baseline (stage_1 forward-only ckpt)"
python3 generate_public_rollout.py --config "$CFG" --checkpoint-dir "$STAGE1_CKPT" \
  --stage-name stage_1 --output-dir artifacts/public_eval_baseline --num-episodes 4
python3 public_eval.py --config "$CFG" \
  --rollout-npz artifacts/public_eval_baseline/rollout_public_eval.npz \
  --output-json artifacts/public_eval_baseline/public_eval.json

echo "==> public benchmark on ours (stage_2 multi-direction ckpt)"
python3 generate_public_rollout.py --config "$CFG" --checkpoint-dir "$STAGE2_CKPT" \
  --stage-name stage_2 --output-dir artifacts/public_eval_bundle --num-episodes 4 --render-first-episode
python3 public_eval.py --config "$CFG" \
  --rollout-npz artifacts/public_eval_bundle/rollout_public_eval.npz \
  --output-json artifacts/public_eval_bundle/public_eval.json

echo "==> per-direction eval on baseline"
python3 per_direction_eval.py --config "$CFG" \
  --checkpoint-dir "$STAGE1_CKPT" --stage-name stage_1 \
  --output-dir artifacts/per_direction_baseline --magnitudes 0.3 0.6 0.9

echo "==> per-direction eval on ours"
python3 per_direction_eval.py --config "$CFG" \
  --checkpoint-dir "$STAGE2_CKPT" --stage-name stage_2 \
  --output-dir artifacts/per_direction_eval --magnitudes 0.3 0.6 0.9

echo "==> render qualitative demo video"
python3 test_policy.py --config "$CFG" \
  --checkpoint-dir "$STAGE2_CKPT" --stage-name stage_2 \
  --output-dir artifacts/demo_bundle

echo "==> generate figures"
python3 make_report_figures.py --artifacts-dir artifacts --output-dir artifacts/figures

echo "[run_all_evals] done"
