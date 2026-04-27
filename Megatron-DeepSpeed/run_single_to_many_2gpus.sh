#!/bin/bash
set -euo pipefail

DIR="$(pwd)"
DATETIME="$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${DIR}/runs"

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

rm -rf z1_uni_ckpt_2g/tensorboard
rm -rf z1_uni_ckpt_2g/checkpoints

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"  # run from repo root per UC README [page:0]

convert_to_universal () {
  local CKPT_ROOT="$1"
  local STEP_TAG="$2"

  rm -rf "${CKPT_ROOT}/${STEP_TAG}_universal"

  python "${HOME}/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py" \
    --input_folder  "${CKPT_ROOT}/${STEP_TAG}" \
    --output_folder "${CKPT_ROOT}/${STEP_TAG}_universal" \
    --inject_missing_state

  # UC README shows latest_universal living alongside global_step*_universal in the same folder. [page:0]
  echo "${STEP_TAG}" > "${CKPT_ROOT}/latest"
  echo "${STEP_TAG}_universal" > "${CKPT_ROOT}/latest_universal"
}

# ================================================================
# ZeRO-1 SOURCE on 2 GPUs
# pick a valid 2-GPU topology. Here: TP=2, PP=1, DP=1.
# ================================================================
Z1_SOURCE_NAME="source_tp2_pp1_dp1_sp1_z1"
Z1_SOURCE_CKPT="${BASE_CHECKPOINT_DIR}/${Z1_SOURCE_NAME}"
Z1_SOURCE_TB="${BASE_LOG_DIR}/${Z1_SOURCE_NAME}"

cat > ds_config_source_z1_2g.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4,
  "steps_per_print": 1,
  "zero_optimization": { "stage": 1 },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT

echo ">>> [2GPU][PHASE 1] Train ZeRO-1 source (TP=2, PP=1, DP=1) ..."
deepspeed --num_gpus 2 pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --train-iters 200 \
  --eval-interval 10 \
  --save "${Z1_SOURCE_CKPT}" \
  --micro-batch-size 4 \
  --data-path "${DATA_PATH}" \
  --vocab-file data/gpt2-vocab.json \
  --merge-file data/gpt2-merges.txt \
  --tensorboard-dir "${Z1_SOURCE_TB}" \
  --deepspeed --deepspeed_config ds_config_source_z1_2g.json \
  --zero-stage 1 \
  ${COMMON_ARGS} \
  | tee "runs/${Z1_SOURCE_NAME}_${DATETIME}.log"

echo ">>> [2GPU][PHASE 1] Convert step100 -> universal..."
convert_to_universal "${Z1_SOURCE_CKPT}" "global_step100"

# ================================================================
# ZeRO-1 TARGETS on 2 GPUs (all valid TP/PP combos on 2 GPUs)
# ================================================================
TARGET_CONFIGS_Z1_2G=(
  "2 1"  # TP=2, PP=1 => DP=1
  "1 2"  # TP=1, PP=2 => DP=1
  "1 1"  # TP=1, PP=1 => DP=2
)

cat > ds_config_target_z1_2g.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 2,
  "steps_per_print": 1,
  "zero_optimization": { "stage": 1 },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT
NUM_GPUS=2
SP=1
for cfg in "${TARGET_CONFIGS_Z1_2G[@]}"; do
  read -r TP PP <<< "$cfg"
  # DP is implied by world size = 2: DP = 2/(TP*PP)
  DP=$(( NUM_GPUS / (TP * PP) ))

  TARGET_NAME="target_tp${TP}_pp${PP}_dp${DP}_sp${SP}_z1"
  TARGET_CKPT="${BASE_CHECKPOINT_DIR}/${TARGET_NAME}"
  TARGET_TB="${BASE_LOG_DIR}/${TARGET_NAME}"

  echo ">>> [2GPU][PHASE 2] Resume ZeRO-1 target ${TARGET_NAME}..."
  deepspeed --num_gpus 2 pretrain_gpt.py \
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
    --deepspeed --deepspeed_config ds_config_target_z1_2g.json \
    --zero-stage 1 \
    ${COMMON_ARGS} \
    | tee "runs/${TARGET_NAME}_${DATETIME}.log"
done

# ================================================================
# EXTRA TARGET: TP=1, PP=1, DP=2, SP=2 (DeepSpeed-Ulysses), ZeRO-1
# world_size=2 => DP = 2/(1*1*2) = 1 "data-parallel" group, but you still run 2 GPUs.
# ================================================================
TP=1
PP=1
DP=2
SP=2
Z=1
NUM_GPUS=2

TARGET_NAME="target_tp${TP}_pp${PP}_dp${DP}_sp${SP}_z${Z}"
TARGET_CKPT="${BASE_CHECKPOINT_DIR}/${TARGET_NAME}"
TARGET_TB="${BASE_LOG_DIR}/${TARGET_NAME}"

echo ">>> [2GPU][PHASE 2] Resume ZeRO-1 target ${TARGET_NAME} (DeepSpeed SP=2)..."
deepspeed --num_gpus ${NUM_GPUS} pretrain_gpt.py \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \
  --ds-sequence-parallel-size ${SP} \
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
  --deepspeed --deepspeed_config ds_config_target_z1_2g.json \
  --zero-stage 1 \
  ${COMMON_ARGS} \
  | tee "runs/${TARGET_NAME}_${DATETIME}.log"
