#!/bin/bash
set -e

DIR=$(pwd)
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$DIR/runs"

# --- CONFIG (Target) ---
TP=2
PP=2
DP=1
LOAD_PATH="z1_uni_ckpt/checkpoints/gpt2/source_bf16_dp4"
SAVE_PATH="z1_uni_ckpt/checkpoints/gpt2/target_bf16_tp2pp2"
# NEW: Define where to save logs
TB_DIR="z1_uni_ckpt/tensorboard/bf16/target_tp2_pp2_dp1_sp1"

rm -rf $SAVE_PATH
rm -rf $TB_DIR

# DeepSpeed Config
DS_CONFIG="ds_config_bf16_target.json"
cat <<EOT > $DS_CONFIG
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 2,
  "steps_per_print": 1,
  "zero_optimization": { "stage": 1 },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOT

echo ">>> STEP 3: Resuming Target..."

deepspeed --num_gpus 4 pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers 4 \
    --hidden-size 512 \
    --num-attention-heads 32 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --train-iters 200 \
    --lr 6.0e-3 \
    --min-lr 6.0e-4 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 10 \
    --data-path ${DIR}/data/my-gpt2_text_document \
    --vocab-file data/gpt2-vocab.json \
    --merge-file data/gpt2-merges.txt \
    --save-interval 100 \
    --load $LOAD_PATH \
    --save $SAVE_PATH \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --bf16 \
    --universal-checkpoint \
    --override-opt_param-scheduler \
    --checkpoint-activations \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --deepspeed \
    --deepspeed_config $DS_CONFIG \
    --zero-stage=1 \
    --deepspeed-activation-checkpointing \
    | tee runs/run_bf16_universal_$DATETIME.log