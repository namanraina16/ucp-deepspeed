
#!/bin/bash
set -euo pipefail

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

# -----------------------------------------------------------------
# Helper: convert a given ZeRO checkpoint tag dir to universal and set latest files
# -----------------------------------------------------------------
convert_to_universal () {
  local CKPT_ROOT="$1"
  local STEP_TAG="$2"

  rm -rf "${CKPT_ROOT}/${STEP_TAG}_universal"

  python "${HOME}/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py" \
    --input_folder  "${CKPT_ROOT}/${STEP_TAG}" \
    --output_folder "${CKPT_ROOT}/${STEP_TAG}_universal" \
    --inject_missing_state

  echo "${STEP_TAG}" > "${CKPT_ROOT}/latest"
  echo "${STEP_TAG}_universal" > "${CKPT_ROOT}/latest_universal"
}

# =================================================================
# PHASE 1: ZeRO-1 SOURCE (TP=2, PP=1, DP=2, SP=1) + convert
# =================================================================
Z1_SOURCE_NAME="source_tp2_pp1_dp2_sp1_z1"
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

echo ">>> [PHASE 1] Train ZeRO-1 source (TP=2, PP=1, DP=2)..."
deepspeed --num_gpus 4 pretrain_gpt.py \
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
  --deepspeed --deepspeed_config ds_config_source_z1.json \
  --zero-stage 1 \
  $COMMON_ARGS \
  | tee "runs/${Z1_SOURCE_NAME}_${DATETIME}.log"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo ">>> [PHASE 1] Convert step100 -> universal..."
test -d "${Z1_SOURCE_CKPT}/global_step100" || { echo "Missing ${Z1_SOURCE_CKPT}/global_step100"; exit 1; }
convert_to_universal "${Z1_SOURCE_CKPT}" "global_step100"

echo ">>> [PHASE 1] Universal contents:"
ls -lah "${Z1_SOURCE_CKPT}/global_step100_universal" | head



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


# Resume from the ZeRO-1 source folder that contains latest_universal
Z1_SOURCE_NAME="source_tp2_pp1_dp2_sp1_z1"
Z1_SOURCE_CKPT="${BASE_CHECKPOINT_DIR}/${Z1_SOURCE_NAME}"

cat > ds_config_target_z3.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 2,
  "steps_per_print": 1,
  "zero_optimization": { "stage": 3 },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT

# ZeRO-3 targets (4 GPUs)
# Fig.7(a) includes TP=1,PP=1 with DP=4 under ZeRO-3.
TARGET_CONFIGS_Z3=(
  "1 1 4"
)
LOAD_TAG="$(cat "${Z1_SOURCE_CKPT}/latest_universal")"
LOAD_DIR="${Z1_SOURCE_CKPT}/${LOAD_TAG}"


for cfg in "${TARGET_CONFIGS_Z3[@]}"; do
  read -r TP PP DP <<< "$cfg"
  TARGET_NAME="target_tp${TP}_pp${PP}_dp${DP}_sp1_z3"
  TARGET_CKPT="${BASE_CHECKPOINT_DIR}/${TARGET_NAME}"
  TARGET_TB="${BASE_LOG_DIR}/${TARGET_NAME}"

  echo ">>> Resume ZeRO-3 target ${TARGET_NAME} from ${Z1_SOURCE_NAME} universal..."
  deepspeed --num_gpus 4 pretrain_gpt.py \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --no-pipeline-parallel \
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
    --deepspeed --deepspeed_config ds_config_target_z3.json \
    --zero-stage 3 \
    $COMMON_ARGS \
    | tee "runs/${TARGET_NAME}_${DATETIME}.log"
done
