#!/bin/bash
set -euo pipefail

# =================================================================
# GLOBAL CONFIGURATION
# =================================================================
DIR=$(pwd)
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$DIR/runs"

DATA_PATH="${DIR}/data/my-gpt2_text_document"
BASE_CHECKPOINT_DIR="z1_uni_ckpt/checkpoints"
BASE_LOG_DIR="z1_uni_ckpt/tensorboard/bf16"

COMMON_ARGS="
  --num-layers 4 --hidden-size 512 --num-attention-heads 32
  --seq-length 512 --max-position-embeddings 512
  --lr 6.0e-3 --min-lr 6.0e-4 --lr-decay-style cosine
  --split 98,2,0 --clip-grad 1.0 --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006
  --bf16 --checkpoint-activations
  --deepspeed-activation-checkpointing
  --tensorboard-log-interval 1
  --eval-iters 40
  --save-interval 100
  --make-vocab-size-divisible-by 256
  --global-batch-size 16
"

echo ">>> cleaning up old tensorboard logs..."
rm -rf z1_uni_ckpt/tensorboard
# rm -rf z1_uni_ckpt/checkpoints   # keep unless full reset desired

# Make sure ds_to_universal can import megatron if needed.
# README also notes scripts should be run from repo root. [page:0]
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Helper: convert a given ZeRO checkpoint tag dir to universal and set latest files
convert_to_universal () {
  local CKPT_ROOT="$1"     # e.g., z1_uni_ckpt/checkpoints/source_...
  local STEP_TAG="$2"      # e.g., global_step100

  rm -rf "${CKPT_ROOT}/${STEP_TAG}_universal"

  python "${HOME}/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py" \
    --input_folder  "${CKPT_ROOT}/${STEP_TAG}" \
    --output_folder "${CKPT_ROOT}/${STEP_TAG}_universal" \
    --inject_missing_state

  echo "${STEP_TAG}" > "${CKPT_ROOT}/latest"
  echo "${STEP_TAG}_universal" > "${CKPT_ROOT}/latest_universal"
}

# =================================================================
# PHASE 1A: ZeRO-1 SOURCE (TP=2, PP=2, DP=1) + convert
# =================================================================
Z1_SOURCE_NAME="source_tp2_pp2_dp1_sp1_z1"
Z1_SOURCE_CKPT="${BASE_CHECKPOINT_DIR}/${Z1_SOURCE_NAME}"
Z1_SOURCE_TB="${BASE_LOG_DIR}/${Z1_SOURCE_NAME}"

cat > ds_config_source_z1.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4,
  "steps_per_print": 1,
  "zero_optimization": { "stage": 1 },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT

echo ">>> [PHASE 1A] Train ZeRO-1 source (TP=2, PP=2, DP=1)..."
deepspeed --num_gpus 4 pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  --train-iters 200 \
  --eval-interval 10 \
  --save "${Z1_SOURCE_CKPT}" \
  --micro-batch-size 4 \
  --data-path "${DATA_PATH}" \
  --vocab-file data/gpt2-vocab.json \
  --merge-file data/gpt2-merges.txt \
  --tensorboard-dir "${Z1_SOURCE_TB}" \
  --deepspeed --deepspeed_config ds_config_source_z1.json \
  --zero-stage 1 \
  $COMMON_ARGS \
  | tee "runs/${Z1_SOURCE_NAME}_${DATETIME}.log"

echo ">>> [PHASE 1A] Convert ZeRO-1 step100 -> universal..."
convert_to_universal "${Z1_SOURCE_CKPT}" "global_step100"

# =================================================================
# PHASE 2A: ZeRO-1 TARGETS (resume from ZeRO-1 universal) - all valid 4-GPU combos
# =================================================================
TARGET_CONFIGS_Z1=(
  "2 2 1"
  "2 1 2"
  "1 2 2"
  "1 1 4"
)

cat > ds_config_target_z1.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 2,
  "steps_per_print": 1,
  "zero_optimization": { "stage": 1 },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT

for cfg in "${TARGET_CONFIGS_Z1[@]}"; do
  read -r TP PP DP <<< "$cfg"
  TARGET_NAME="target_tp${TP}_pp${PP}_dp${DP}_sp1_z1"
  TARGET_CKPT="${BASE_CHECKPOINT_DIR}/${TARGET_NAME}"
  TARGET_TB="${BASE_LOG_DIR}/${TARGET_NAME}"

  echo ">>> [PHASE 2A] Resume ZeRO-1 target ${TARGET_NAME}..."
  deepspeed --num_gpus 4 pretrain_gpt.py \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --train-iters 200 \
    --exit-interval 200 \
    --eval-interval 10 \
    --load "${Z1_SOURCE_CKPT}" \
    --save "${TARGET_CKPT}" \
    --micro-batch-size 2 \
    --universal-checkpoint \
    --override-opt_param-scheduler \
    --data-path "${DATA_PATH}" \
    --vocab-file data/gpt2-vocab.json \
    --merge-file data/gpt2-merges.txt \
    --tensorboard-dir "${TARGET_TB}" \
    --deepspeed --deepspeed_config ds_config_target_z1.json \
    --zero-stage 1 \
    $COMMON_ARGS \
    | tee "runs/${TARGET_NAME}_${DATETIME}.log"
done



# =================================================================
# PHASE 1B: ZeRO-2 SOURCE (must use --no-pipeline-parallel, so PP=1) + convert
# =================================================================
# Choose a valid 4-GPU topology with PP=1. Example: TP=2, PP=1, DP=2.
Z2_SOURCE_NAME="source_tp2_pp1_dp2_sp1_z2"
Z2_SOURCE_CKPT="${BASE_CHECKPOINT_DIR}/${Z2_SOURCE_NAME}"
Z2_SOURCE_TB="${BASE_LOG_DIR}/${Z2_SOURCE_NAME}"

cat > ds_config_source_z2.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true
  },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT


echo ">>> [PHASE 1B] Train ZeRO-2 source (TP=2, PP=1, DP=2) with --no-pipeline-parallel..."
deepspeed --num_gpus 4 pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --no-pipeline-parallel \
  --train-iters 200 \
  --eval-interval 10 \
  --save "${Z2_SOURCE_CKPT}" \
  --micro-batch-size 4 \
  --data-path "${DATA_PATH}" \
  --vocab-file data/gpt2-vocab.json \
  --merge-file data/gpt2-merges.txt \
  --tensorboard-dir "${Z2_SOURCE_TB}" \
  --deepspeed --deepspeed_config ds_config_source_z2.json \
  --zero-stage 2 \
  $COMMON_ARGS \
  | tee "runs/${Z2_SOURCE_NAME}_${DATETIME}.log"

echo ">>> [PHASE 1B] Convert ZeRO-2 step100 -> universal..."
convert_to_universal "${Z2_SOURCE_CKPT}" "global_step100"

echo ">>> Verifying ZeRO-2 universal checkpoint files..."
cat z1_uni_ckpt/checkpoints/source_tp2_pp1_dp2_sp1_z2/latest_universal
ls -l z1_uni_ckpt/checkpoints/source_tp2_pp1_dp2_sp1_z2/$(cat .../latest_universal)

# =================================================================
# PHASE 2B: ZeRO-2 TARGETS (resume from ZeRO-2 universal) - PP must be 1
# =================================================================
TARGET_CONFIGS_Z2=(
  "2 1 2"
  "1 1 4"
)

cat > ds_config_target_z2.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 2,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true
  },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT

for cfg in "${TARGET_CONFIGS_Z2[@]}"; do
  read -r TP PP DP <<< "$cfg"
  TARGET_NAME="target_tp${TP}_pp${PP}_dp${DP}_sp1_z2"
  TARGET_CKPT="${BASE_CHECKPOINT_DIR}/${TARGET_NAME}"
  TARGET_TB="${BASE_LOG_DIR}/${TARGET_NAME}"

  echo ">>> [PHASE 2B] Resume ZeRO-2 target ${TARGET_NAME}..."
  deepspeed --num_gpus 4 pretrain_gpt.py \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --no-pipeline-parallel \
    --train-iters 200 \
    --exit-interval 200 \
    --eval-interval 10 \
    --load "${Z2_SOURCE_CKPT}" \
    --save "${TARGET_CKPT}" \
    --micro-batch-size 2 \
    --universal-checkpoint \
    --override-opt_param-scheduler \
    --data-path "${DATA_PATH}" \
    --vocab-file data/gpt2-vocab.json \
    --merge-file data/gpt2-merges.txt \
    --tensorboard-dir "${TARGET_TB}" \
    --deepspeed --deepspeed_config ds_config_target_z2.json \
    --zero-stage 2 \
    $COMMON_ARGS \
    | tee "runs/${TARGET_NAME}_${DATETIME}.log"
done

# =================================================================
# PHASE 1C: ZeRO-3 SOURCE (README note: universal supports Data Parallelism)
# =================================================================
# README note implies keep TP=1, PP=1 and only vary DP for stage 3 universal. [page:0]
Z3_SOURCE_NAME="source_tp1_pp1_dp4_sp1_z3"
Z3_SOURCE_CKPT="${BASE_CHECKPOINT_DIR}/${Z3_SOURCE_NAME}"
Z3_SOURCE_TB="${BASE_LOG_DIR}/${Z3_SOURCE_NAME}"

cat > ds_config_source_z3.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": 3
  },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT

echo ">>> [PHASE 1C] Train ZeRO-3 source (TP=1, PP=1, DP=4) with --no-pipeline-parallel..."
deepspeed --num_gpus 4 pretrain_gpt.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --no-pipeline-parallel \
  --train-iters 200 \
  --eval-interval 10 \
  --save "${Z3_SOURCE_CKPT}" \
  --micro-batch-size 4 \
  --data-path "${DATA_PATH}" \
  --vocab-file data/gpt2-vocab.json \
  --merge-file data/gpt2-merges.txt \
  --tensorboard-dir "${Z3_SOURCE_TB}" \
  --deepspeed --deepspeed_config ds_config_source_z3.json \
  --zero-stage 3 \
  $COMMON_ARGS \
  | tee "runs/${Z3_SOURCE_NAME}_${DATETIME}.log"

echo ">>> [PHASE 1C] Convert ZeRO-3 step100 -> universal..."
convert_to_universal "${Z3_SOURCE_CKPT}" "global_step100"

# =================================================================
# PHASE 2C: ZeRO-3 TARGETS (DP-only changes; TP=1, PP=1)
# =================================================================
# Example DP change: 4 -> 2 (would require 2 GPUs), but you said fixed 4 GPUs.
# So on 4 GPUs, the only DP=4 with TP=1,PP=1 is "1 1 4" (no change).
# You can still keep this block for completeness; adjust num_gpus if you want DP=2.
TARGET_CONFIGS_Z3=(
  "1 1 4"
)

cat > ds_config_target_z3.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 2,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": 3
  },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT

for cfg in "${TARGET_CONFIGS_Z3[@]}"; do
  read -r TP PP DP <<< "$cfg"
  TARGET_NAME="target_tp${TP}_pp${PP}_dp${DP}_sp1_z3"
  TARGET_CKPT="${BASE_CHECKPOINT_DIR}/${TARGET_NAME}"
  TARGET_TB="${BASE_LOG_DIR}/${TARGET_NAME}"

  echo ">>> [PHASE 2C] Resume ZeRO-3 target ${TARGET_NAME}..."
  deepspeed --num_gpus 4 pretrain_gpt.py \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --no-pipeline-parallel \
    --train-iters 200 \
    --exit-interval 200 \
    --eval-interval 10 \
    --load "${Z3_SOURCE_CKPT}" \
    --save "${TARGET_CKPT}" \
    --micro-batch-size 2 \
    --universal-checkpoint \
    --override-opt_param-scheduler \
    --data-path "${DATA_PATH}" \
    --vocab-file data/gpt2-vocab.json \
    --merge-file data/gpt2-merges.txt \
    --tensorboard-dir "${TARGET_TB}" \
    --deepspeed --deepspeed_config ds_config_target_z3.json \
    --zero-stage 3 \
    $COMMON_ARGS \
    | tee "runs/${TARGET_NAME}_${DATETIME}.log"
done

