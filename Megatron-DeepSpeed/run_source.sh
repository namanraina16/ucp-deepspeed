#!/bin/bash
set -e  # Exit strictly if any command fails

# =================================================================
# GLOBAL CONFIGURATION (Aligned with Official Toy Config)
# =================================================================
DIR=$(pwd)
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$DIR/runs"

# Paths
DATA_PATH="${DIR}/data/my-gpt2_text_document"
BASE_CHECKPOINT_DIR="z1_uni_ckpt/checkpoints"
BASE_LOG_DIR="z1_uni_ckpt/tensorboard/bf16"

# Source Config (TP=1, PP=1, DP=4)
SOURCE_NAME="source_tp1_pp1_dp4_sp1"
SOURCE_CKPT_PATH="${BASE_CHECKPOINT_DIR}/${SOURCE_NAME}"
SOURCE_LOG_DIR="${BASE_LOG_DIR}/${SOURCE_NAME}"

# Clean Start
echo ">>> cleaning up old runs..."
rm -rf z1_uni_ckpt/tensorboard
rm -rf z1_uni_ckpt/checkpoints

# Common Training Flags (Change 1: GBS unified to 4)
COMMON_ARGS="
    --num-layers 4 --hidden-size 512 --num-attention-heads 32
    --seq-length 512 --max-position-embeddings 512
    --lr 6.0e-3 --min-lr 6.0e-4 --lr-decay-style cosine
    --split 98,2,0 --clip-grad 1.0 --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006
    --bf16 --checkpoint-activations
    --zero-stage=1 --deepspeed-activation-checkpointing
    --tensorboard-log-interval 1
    --eval-iters 40
    --save-interval 100
    --make-vocab-size-divisible-by 256
    --global-batch-size 4
"

# =================================================================
# PHASE 1: TRAIN SOURCE (Steps 0 -> 100)
# =================================================================
echo "-----------------------------------------------------------"
echo ">>> [PHASE 1] Training Source Model (TP=1, PP=1, DP=4)..."
echo "-----------------------------------------------------------"

cat <<EOT > ds_config_source.json
{
  "train_batch_size": 4,
  "train_micro_batch_size_per_gpu": 1,
  "steps_per_print": 1,
  "zero_optimization": { "stage": 1 },
  "bf16": { "enabled": true },
  "data_types": { "grad_accum_dtype": "fp32" }
}
EOT

deepspeed --num_gpus 4 pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --train-iters 100 \
    --eval-interval 10 \
    --save $SOURCE_CKPT_PATH \
    --micro-batch-size 1 \
    --data-path $DATA_PATH \
    --vocab-file data/gpt2-vocab.json \
    --merge-file data/gpt2-merges.txt \
    --tensorboard-dir $SOURCE_LOG_DIR \
    --deepspeed --deepspeed_config ds_config_source.json \
    $COMMON_ARGS \
    | tee runs/source_${DATETIME}.log

echo ">>> [PHASE 1] Converting Checkpoint to Universal Format..."
python -m deepspeed.checkpoint.ds_to_universal \
    --input_folder ${SOURCE_CKPT_PATH}/global_step100 \
    --output_folder ${SOURCE_CKPT_PATH}/global_step100_universal

# =================================================================
# PHASE 2: TRAIN ALL TARGETS (Steps 100 -> 200)
# =================================================================
TARGET_CONFIGS=(
    "2 2 1"  # Target A: 3D Parallel
    "2 1 2"  # Target B: Tensor + Data
    "1 2 2"  # Target C: Pipeline + Data
    "4 1 1"  # Target D: Pure Tensor
)

for config in "${TARGET_CONFIGS[@]}"; do
    read -r TP PP DP <<< "$config"
    TARGET_NAME="target_tp${TP}_pp${PP}_dp${DP}_sp1"
    TARGET_CKPT_PATH="${BASE_CHECKPOINT_DIR}/${TARGET_NAME}"
    TARGET_LOG_DIR="${BASE_LOG_DIR}/${TARGET_NAME}"

    echo ">>> [PHASE 2] Launching Target: TP=$TP, PP=$PP, DP=$DP..."

    # Target DS Config (Change 2: Explicit grad_accum_dtype)
    cat <<EOT > ds_config_target.json
    {
      "train_batch_size" : 4,
      "train_micro_batch_size_per_gpu": 1,
      "steps_per_print": 1,
      "zero_optimization": { "stage": 1 },
      "bf16": { "enabled": true },
      "data_types": { "grad_accum_dtype": "fp32" },
      "wall_clock_breakdown" : false
    }
EOT

    # Target Training (Change 3: Exit-Interval added)
    deepspeed --num_gpus 4 pretrain_gpt.py \
        --tensor-model-parallel-size $TP \
        --pipeline-model-parallel-size $PP \
        --train-iters 200 \
        --eval-interval 10 \
        --load $SOURCE_CKPT_PATH \
        --save $TARGET_CKPT_PATH \
        --micro-batch-size 1 \
        --universal-checkpoint \
        --override-opt_param-scheduler \
        --exit-interval 200 \
        --data-path $DATA_PATH \
        --vocab-file data/gpt2-vocab.json \
        --merge-file data/gpt2-merges.txt \
        --tensorboard-dir $TARGET_LOG_DIR \
        --deepspeed --deepspeed_config ds_config_target.json \
        $COMMON_ARGS \
        | tee runs/target_tp${TP}_pp${PP}_${DATETIME}.log
done

echo ">>> ALL EXPERIMENTS COMPLETE. Running Analysis Script..."
bash examples_deepspeed/universal_checkpointing/megatron_gpt/run_tb_analysis_gpt.sh z1_uni_ckpt