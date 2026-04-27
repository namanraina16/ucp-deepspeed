#!/bin/bash
set -euo pipefail

NUM_GPUS=4
ROOT="$(pwd)"
BASE="z2_uni_ckpt"

SRC_CKPT="${BASE}/checkpoints/source_tp1_pp1_dp4_sp1_z2"
STEP_TAG="global_step100"

TGT_CKPT="${BASE}/checkpoints/target_tp1_pp1_dp2_sp1_z2"
TGT_TB="${BASE}/tensorboard/bf16/target_tp1_pp1_dp2_sp1_z2"
DATA_PATH="${ROOT}/data/my-gpt2_text_document"

mkdir -p "${TGT_CKPT}" "${TGT_TB}" runs

# Convert step100 -> universal and create latest_universal (as README expects). [page:8]
test -d "${SRC_CKPT}/${STEP_TAG}" || { echo "Missing ${SRC_CKPT}/${STEP_TAG}"; exit 1; }

rm -rf "${SRC_CKPT}/${STEP_TAG}_universal"
python -m deepspeed.checkpoint.ds_to_universal \
  --input_folder  "${SRC_CKPT}/${STEP_TAG}" \
  --output_folder "${SRC_CKPT}/${STEP_TAG}_universal" \
  --inject_missing_state

echo "${STEP_TAG}" > "${SRC_CKPT}/latest"
echo "${STEP_TAG}_universal" > "${SRC_CKPT}/latest_universal"

cat > ds_z2.json <<'EOT'
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 2,
  "steps_per_print": 1,
  "zero_optimization": { "stage": 2 },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT

COMMON_ARGS="
  --num-layers 4 --hidden-size 512 --num-attention-heads 32
  --seq-length 512 --max-position-embeddings 512
  --lr 6.0e-3 --min-lr 6.0e-4 --lr-decay-style cosine
  --split 98,2,0 --clip-grad 1.0 --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006
  --bf16 --checkpoint-activations
  --deepspeed-activation-checkpointing
  --tensorboard-log-interval 1
  --eval-iters 40 --eval-interval 10
  --save-interval 100
  --make-vocab-size-divisible-by 256
  --global-batch-size 16
"

# DP=2 target: on 4 GPUs, this implies extra non-DP parallelism is 2.
# Choose TP=2,PP=1 so TP*PP*DP=4.
deepspeed --num_gpus ${NUM_GPUS} pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --no-pipeline-parallel \
  --train-iters 200 \
  --load "${SRC_CKPT}" \
  --universal-checkpoint \
  --override-opt_param-scheduler \
  --save "${TGT_CKPT}" \
  --micro-batch-size 2 \
  --data-path "${DATA_PATH}" \
  --vocab-file data/gpt2-vocab.json \
  --merge-file data/gpt2-merges.txt \
  --tensorboard-dir "${TGT_TB}" \
  --deepspeed --deepspeed_config ds_z2.json \
  --zero-stage 2 \
  ${COMMON_ARGS} | tee "runs/z2_target_${TGT_CKPT//\//_}.log"
