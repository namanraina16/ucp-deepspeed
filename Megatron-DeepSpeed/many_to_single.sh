#!/bin/bash
set -euo pipefail

NUM_GPUS=4
DATA_PATH="$(pwd)/data/my-gpt2_text_document"

SAVE_STEP=200
TRAIN_ITERS_SOURCE=200
RESUME_ITERS_TARGET=300

SEED=1234

ROOT="$(pwd)"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUNS_DIR="${ROOT}/runs"
mkdir -p "${RUNS_DIR}"

BASE_CKPT="${ROOT}/ucp_ckpt/checkpoints"
BASE_TB="${ROOT}/ucp_ckpt/tensorboard/bf16"
mkdir -p "${BASE_CKPT}" "${BASE_TB}"

# ---------- Preflight checks ----------
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

enforce_zero_for_pp () {
  local PP="$1"
  local ZERO="$2"
  if [[ "${PP}" -gt 1 && "${ZERO}" -ge 2 ]]; then
    echo "ERROR: PP=${PP} with ZeRO=${ZERO} is invalid here; use ZeRO-1." 1>&2
    exit 1
  fi
}

# Target: TP=2, PP=2, DP=1, SP=1
TARGET_NAME_BASE="target_tp2_pp2_dp1_sp1"
TARGET_TP=2
TARGET_PP=2
TARGET_DP=1
TARGET_MICRO_BS=2
TARGET_ZERO_STAGE=1

# ----- sources -----
# IMPORTANT: For UCP in Megatron-DeepSpeed, do NOT mix PP=1 (--no-pipeline-parallel)
# with PP>1 runs when converting/loading. Keep PP consistent with target. [page:1]
SOURCES=(
  "src_tp2_pp2_dp1_sp1_z1  2 2 1  1  4"
  "src_tp1_pp2_dp2_sp1_z1  1 2 2  1  4"
  "src_tp2_pp2_dp2_sp1_z1  2 2 2  1  4"
)

for entry in "${SOURCES[@]}"; do
  read -r SRC_NAME SRC_TP SRC_PP SRC_DP SRC_ZERO SRC_MBS <<< "${entry}"

  enforce_zero_for_pp "${SRC_PP}" "${SRC_ZERO}"
  enforce_zero_for_pp "${TARGET_PP}" "${TARGET_ZERO_STAGE}"

  # Guard against unsupported pipeline-mode conversion. [page:1]
  if [[ "${SRC_PP}" -eq 1 && "${TARGET_PP}" -gt 1 ]]; then
    echo "ERROR: Unsupported: PP=1 source (no-pipeline) -> PP>1 target (pipeline)." 1>&2
    exit 1
  fi
  if [[ "${SRC_PP}" -gt 1 && "${TARGET_PP}" -eq 1 ]]; then
    echo "ERROR: Unsupported: PP>1 source (pipeline) -> PP=1 target (no-pipeline)." 1>&2
    exit 1
  fi

  SRC_CKPT="${BASE_CKPT}/${SRC_NAME}"
  SRC_TB="${BASE_TB}/${SRC_NAME}"
  SRC_DS_JSON="${ROOT}/ds_${SRC_NAME}.json"
  write_ds_config "${SRC_DS_JSON}" "${SRC_ZERO}" 16 "${SRC_MBS}"

  # pipeline args: since SRC_PP=2, this stays empty (no --no-pipeline-parallel)
  SRC_PIPE_ARGS=()
  if [[ "${SRC_PP}" -eq 1 ]]; then
    SRC_PIPE_ARGS+=(--no-pipeline-parallel)
  fi

  echo "=== [SOURCE] ${SRC_NAME} (TP=${SRC_TP} PP=${SRC_PP} DP=${SRC_DP} ZeRO=${SRC_ZERO}) ==="
  deepspeed --num_gpus "${NUM_GPUS}" pretrain_gpt.py \
    --tensor-model-parallel-size "${SRC_TP}" \
    --pipeline-model-parallel-size "${SRC_PP}" \
    "${SRC_PIPE_ARGS[@]}" \
    --train-iters "${TRAIN_ITERS_SOURCE}" \
    --save "${SRC_CKPT}" \
    --tensorboard-dir "${SRC_TB}" \
    --micro-batch-size "${SRC_MBS}" \
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
  test -d "${UCP_DIR}" || { echo "Missing universal dir: ${UCP_DIR}" 1>&2; exit 1; }
  test -d "${UCP_DIR}/zero" || { echo "Missing universal zero dir: ${UCP_DIR}/zero" 1>&2; exit 1; }
  ls "${UCP_DIR}/zero" | head -n 1 >/dev/null || { echo "Universal zero dir empty: ${UCP_DIR}/zero" 1>&2; exit 1; }

  TARGET_NAME="${TARGET_NAME_BASE}_z${TARGET_ZERO_STAGE}"
  TARGET_DS_JSON="${ROOT}/ds_${TARGET_NAME}.json"
  write_ds_config "${TARGET_DS_JSON}" "${TARGET_ZERO_STAGE}" 16 "${TARGET_MICRO_BS}"

  RESUME_NAME="${SRC_NAME}__to__${TARGET_NAME}"
  RESUME_CKPT="${BASE_CKPT}/${RESUME_NAME}"
  RESUME_TB="${BASE_TB}/${RESUME_NAME}"

  TGT_PIPE_ARGS=()  # PP=2, so no --no-pipeline-parallel

  echo "=== [RESUME->TARGET] ${RESUME_NAME} (TP=${TARGET_TP} PP=${TARGET_PP} DP=${TARGET_DP} SP=1 ZeRO=${TARGET_ZERO_STAGE}) ==="
  deepspeed --num_gpus "${NUM_GPUS}" pretrain_gpt.py \
    --tensor-model-parallel-size "${TARGET_TP}" \
    --pipeline-model-parallel-size "${TARGET_PP}" \
    "${TGT_PIPE_ARGS[@]}" \
    --train-iters "${RESUME_ITERS_TARGET}" \
    --load "${SRC_CKPT}" \
    --save "${RESUME_CKPT}" \
    --tensorboard-dir "${RESUME_TB}" \
    --micro-batch-size "${TARGET_MICRO_BS}" \
    --universal-checkpoint \
    --override-opt_param-scheduler \
    --deepspeed --deepspeed_config "${TARGET_DS_JSON}" \
    "${COMMON_ARGS[@]}" \
    | tee "${RUNS_DIR}/${RESUME_NAME}_${TS}.log"
done

echo "DONE. Logs in ${RUNS_DIR} and TB under ${BASE_TB}."
