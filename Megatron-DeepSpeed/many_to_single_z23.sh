#!/bin/bash
set -euo pipefail

# ---------------- knobs ----------------
NUM_GPUS=4                    # for source runs (DP=4 when TP=PP=1)
DATA_PATH="$(pwd)/data/my-gpt2_text_document"

SAVE_STEP=200                 # set to 200 if you want resume at 200
TRAIN_ITERS_SOURCE=200
RESUME_ITERS_TARGET=300

SEED=1234

ROOT="$(pwd)"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUNS_DIR="${ROOT}/runs_zero23"
mkdir -p "${RUNS_DIR}"

BASE_CKPT="${ROOT}/ucp_ckpt_zero23/checkpoints"
BASE_TB="${ROOT}/ucp_ckpt_zero23/tensorboard/bf16"
mkdir -p "${BASE_CKPT}" "${BASE_TB}"

# ---------- Preflight ----------
test -f "${DATA_PATH}.bin" -a -f "${DATA_PATH}.idx" || {
  echo "ERROR: Missing indexed dataset: ${DATA_PATH}.bin/.idx" 1>&2
  exit 1
}

python - <<'PY'
import importlib
m = importlib.import_module("deepspeed.checkpoint.ds_to_universal")
print("OK: deepspeed.checkpoint.ds_to_universal importable:", m.__name__)
PY

COMMON_ARGS=(
  --num-layers 4 --hidden-size 512 --num-attention-heads 32
  --seq-length 512 --max-position-embeddings 512
  --lr 6.0e-3 --min-lr 6.0e-4 --lr-decay-style cosine
  --split 98,2,0 --clip-grad 1.0 --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006
  --bf16 --checkpoint-activations
  --deepspeed-activation-checkpointing
  --tensorboard-log-interval 1
  --eval-iters 40
  --eval-interval 10
  --save-interval "${SAVE_STEP}"
  --make-vocab-size-divisible-by 256
  --global-batch-size 16
  --data-path "${DATA_PATH}"
  --vocab-file data/gpt2-vocab.json
  --merge-file data/gpt2-merges.txt
  --seed "${SEED}"
  --ds-sequence-parallel-size 1
)

write_ds_config () {
  local OUT="$1"
  local ZERO_STAGE="$2"
  local TRAIN_BS="$3"
  local MBS="$4"
  cat > "${OUT}" <<EOT
{
  "train_batch_size": ${TRAIN_BS},
  "train_micro_batch_size_per_gpu": ${MBS},
  "steps_per_print": 1,
  "zero_optimization": { "stage": ${ZERO_STAGE} },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT
}

convert_to_universal () {
  local CKPT_ROOT="$1"
  local STEP_TAG="$2"

  rm -rf "${CKPT_ROOT}/${STEP_TAG}_universal"
  python -m deepspeed.checkpoint.ds_to_universal \
    --input_folder  "${CKPT_ROOT}/${STEP_TAG}" \
    --output_folder "${CKPT_ROOT}/${STEP_TAG}_universal" \
    --inject_missing_state

  echo "${STEP_TAG}" > "${CKPT_ROOT}/latest"
  echo "${STEP_TAG}_universal" > "${CKPT_ROOT}/latest_universal"
}

# ---------- fixed parallelism for ZeRO-2/3 ----------
# UCP README: ZeRO-2/3 require --no-pipeline-parallel. [page:1]
TP=1
PP=1
SP=1
PIPE_ARGS=(--no-pipeline-parallel)

# For Fig7-like DP change: source DP=4 -> target DP=2 (same model, different world size)
SRC_GPUS=4
TGT_GPUS=2

# micro-batch size (keep constant)
MBS=2

run_zero_stage () {
  local Z="$1"   # 2 or 3

  local SRC_NAME="src_tp${TP}_pp${PP}_dp${SRC_GPUS}_sp${SP}_z${Z}"
  local TGT_NAME="target_tp${TP}_pp${PP}_dp${TGT_GPUS}_sp${SP}_z${Z}"
  local RESUME_NAME="${SRC_NAME}__to__${TGT_NAME}"

  local SRC_CKPT="${BASE_CKPT}/${SRC_NAME}"
  local SRC_TB="${BASE_TB}/${SRC_NAME}"
  local SRC_DS_JSON="${ROOT}/ds_${SRC_NAME}.json"

  local TGT_CKPT="${BASE_CKPT}/${RESUME_NAME}"
  local TGT_TB="${BASE_TB}/${RESUME_NAME}"
  local TGT_DS_JSON="${ROOT}/ds_${TGT_NAME}.json"

  write_ds_config "${SRC_DS_JSON}" "${Z}" 16 "${MBS}"
  write_ds_config "${TGT_DS_JSON}" "${Z}" 16 "${MBS}"

  echo "=== [SOURCE ZeRO-${Z}] ${SRC_NAME} ==="
  deepspeed --num_gpus "${SRC_GPUS}" pretrain_gpt.py \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    "${PIPE_ARGS[@]}" \
    --train-iters "${TRAIN_ITERS_SOURCE}" \
    --save "${SRC_CKPT}" \
    --tensorboard-dir "${SRC_TB}" \
    --micro-batch-size "${MBS}" \
    --deepspeed --deepspeed_config "${SRC_DS_JSON}" \
    "${COMMON_ARGS[@]}" \
    | tee "${RUNS_DIR}/${SRC_NAME}_${TS}.log"

  echo "=== [CONVERT] ${SRC_NAME} global_step${SAVE_STEP} -> universal ==="
  test -d "${SRC_CKPT}/global_step${SAVE_STEP}" || {
    echo "Missing ${SRC_CKPT}/global_step${SAVE_STEP}" 1>&2
    exit 1
  }
  convert_to_universal "${SRC_CKPT}" "global_step${SAVE_STEP}"

  UCP_DIR="${SRC_CKPT}/global_step${SAVE_STEP}_universal"
  test -d "${UCP_DIR}/zero" || { echo "Missing ${UCP_DIR}/zero" 1>&2; exit 1; }

  echo "=== [RESUME->TARGET ZeRO-${Z}] ${RESUME_NAME} ==="
  deepspeed --num_gpus "${TGT_GPUS}" pretrain_gpt.py \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    "${PIPE_ARGS[@]}" \
    --train-iters "${RESUME_ITERS_TARGET}" \
    --load "${SRC_CKPT}" \
    --save "${TGT_CKPT}" \
    --tensorboard-dir "${TGT_TB}" \
    --micro-batch-size "${MBS}" \
    --universal-checkpoint \
    --override-opt_param-scheduler \
    --deepspeed --deepspeed_config "${TGT_DS_JSON}" \
    "${COMMON_ARGS[@]}" \
    | tee "${RUNS_DIR}/${RESUME_NAME}_${TS}.log"
}

run_zero_stage 2
run_zero_stage 3

echo "DONE. Logs in ${RUNS_DIR} and TB under ${BASE_TB}."
