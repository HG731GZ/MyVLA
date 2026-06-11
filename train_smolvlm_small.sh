#!/bin/bash
# SimVLA Training Script for LIBERO (Small Model)
# 
# Key features:
#   - 384x384 image resolution (SmolVLM requirement)
#   - All views processed together by VLM (no aux_visual_inputs)
#   - Smaller action transformer configuration

set -e

# =============================================================================
# Command line arguments (with defaults)
# =============================================================================

BATCH_SIZE=${1:-64}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_libero_small}
RESUME_CKPT=${4:-""}
GRADIENT_ACCUMULATION_STEPS=${5:-${GRADIENT_ACCUMULATION_STEPS:-1}}

echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"
echo "   gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}"

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Suppress TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONNOUSERSITE=1

# =============================================================================
# Path configuration
# =============================================================================
LIBERO_DATA_DIR=${LIBERO_DATA_DIR:-./datasets/metas}
NORM_STATS_PATH=${NORM_STATS_PATH:-./norm_stats/libero_object_norm.json}
TRAIN_METAS_PATH=${TRAIN_METAS_PATH:-./datasets/metas/libero_object_train.json}
LIBERO_SUBSETS=${LIBERO_SUBSETS:-libero_object}
LIBERO_STRICT_VALIDATION=${LIBERO_STRICT_VALIDATION:-1}
LIBERO_FORCE_REBUILD=${LIBERO_FORCE_REBUILD:-0}
LIBERO_EXCLUDE_FILES=${LIBERO_EXCLUDE_FILES:-}

# SmolVLM backbone (can be local path or HuggingFace repo)
SMOLVLM_MODEL=${SMOLVLM_MODEL:-./pretrained/SmolVLM-500M-Instruct}

# =============================================================================
# Training hyperparameters
# =============================================================================
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_ACTIONS=${NUM_ACTIONS:-10}
ITERS=${ITERS:-800000}
WARMUP_STEPS=${WARMUP_STEPS:-0}
FREEZE_VLM_STEPS=${FREEZE_VLM_STEPS:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-20000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
NUM_PROCESSES=${NUM_PROCESSES:-4}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29504}
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))
VLM_TORCH_DTYPE=${VLM_TORCH_DTYPE:-float32}
NUM_VIEWS=2
LANGUAGE_MAX_LENGTH=96

# Model architecture (Small configuration)
HIDDEN_SIZE=768         
DEPTH=9
NUM_HEADS=12             
ATTENTION_DROPOUT=0.1

# =============================================================================
# Step 1: Validate dataset and refresh derived data files
# =============================================================================
read -r -a LIBERO_SUBSET_ARGS <<< "$LIBERO_SUBSETS"
PREPARE_ARGS=(
    python prepare_libero_data.py
    --data_dir "$LIBERO_DATA_DIR"
    --subsets "${LIBERO_SUBSET_ARGS[@]}"
    --metadata_output "$TRAIN_METAS_PATH"
    --norm_stats_output "$NORM_STATS_PATH"
)
if [ -n "$LIBERO_EXCLUDE_FILES" ]; then
    read -r -a LIBERO_EXCLUDE_FILE_ARGS <<< "$LIBERO_EXCLUDE_FILES"
    for exclude_pattern in "${LIBERO_EXCLUDE_FILE_ARGS[@]}"; do
        PREPARE_ARGS+=(--exclude_file "$exclude_pattern")
    done
fi

case "${LIBERO_STRICT_VALIDATION,,}" in
    1|true|yes|on)
        echo "LIBERO completeness validation: strict"
        ;;
    0|false|no|off)
        echo "LIBERO completeness validation: relaxed (leave-out experiment)"
        PREPARE_ARGS+=(--allow_incomplete)
        ;;
    *)
        echo "ERROR: LIBERO_STRICT_VALIDATION must be 1/0 or true/false"
        exit 2
        ;;
esac

case "${LIBERO_FORCE_REBUILD,,}" in
    1|true|yes|on)
        PREPARE_ARGS+=(--force_rebuild)
        ;;
    0|false|no|off)
        ;;
    *)
        echo "ERROR: LIBERO_FORCE_REBUILD must be 1/0 or true/false"
        exit 2
        ;;
esac

"${PREPARE_ARGS[@]}"

# =============================================================================
# Step 2: Build training arguments
# =============================================================================
ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode libero_joint \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_vlm_steps ${FREEZE_VLM_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --vlm_torch_dtype ${VLM_TORCH_DTYPE} \
    --num_views ${NUM_VIEWS} \
    --language_max_length ${LANGUAGE_MAX_LENGTH} \
    --attention_dropout ${ATTENTION_DROPOUT}"

# Add resume checkpoint if specified
if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# Step 3: Start training
# =============================================================================
echo "============================================================"
echo "Starting SimVLA Training on LIBERO (Small Action Transformer)"
echo "============================================================"
echo "SmolVLM backbone: ${SMOLVLM_MODEL}"
echo "Data directory: $LIBERO_DATA_DIR"
echo "Normalization stats: $NORM_STATS_PATH"
echo "Action mode: libero_joint"
echo "Batch size: ${BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective global batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Learning coef: ${LEARNING_COEF}"
echo "Num actions: ${NUM_ACTIONS}"
echo "Image size: 384x384"
echo "============================================================"
echo "Action Transformer configuration:"
echo "   Hidden size: ${HIDDEN_SIZE}"
echo "   Depth: ${DEPTH}"
echo "   Num heads: ${NUM_HEADS}"
echo "   Action head: CrossAttention -> SelfAttention -> MLP"
echo "   Num views: ${NUM_VIEWS} (AgentView + WristView)"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

# Multi-GPU training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port ${MAIN_PROCESS_PORT} \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
