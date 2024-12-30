#!/bin/bash

# Output model directory
output_model=./demo/output

# Create the output directory if it doesn't exist
if [ ! -d "${output_model}" ]; then
    mkdir -p "${output_model}"
fi

# Copy necessary files to the output directory
cp ./pretrain.sh "${output_model}"
cp ./config/ds_config_zero*.json "${output_model}"
cp ./model/configuration*.py "${output_model}"
cp ./model/modeling*.py "${output_model}"
cp ./model/tokenization*.py "${output_model}"

# Set environment variables
export PYTHONPATH="./:$(pwd):${PYTHONPATH}"
export DS_ENV_FILE="./config/.deepspeed_env"
export TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX"
export CUTLASS_PATH="/opt/cutlass"
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

# Run the training script with DeepSpeed
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    --master_port 22001 \
    pretrain.py \
    --task_type classifacation \
    --train_dir ./data/processed/train \
    --validation_dir ./data/processed/validation \
    --cache_dir ./data/cache \
    --streaming false \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir "${output_model}" \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --max_train_samples 1000 \
    --warmup_steps 100 \
    --preprocessing_num_workers 128 \
    --logging_dir "${output_model}/logs" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --log_level "info" \
    --save_strategy "steps" \
    --save_steps 100 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_total_limit 5 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 4096 \
    --step_size 4096 \
    --patch_size 1 \
    --overwrite_output_dir true \
    --run_name "${output_model}" \
    --bf16 \
    --bf16_full_eval \
    --load_best_model_at_end true \
    --metric_for_best_model loss \
    --greater_is_better false \
    --gradient_checkpointing \
    --deepspeed ./config/ds_config_zero1.json \
    --ignore_data_skip false \
    --ddp_timeout 18000000 \
    | tee -a "${output_model}/pretrain.log"
    # > "${output_model}/pretrain.log" 2>&1

# Optional commented-out parameters
# --number_classes 2 \
# --resume_from_checkpoint ${output_model}/checkpoint-20400 \
# --max_steps 100 \
# --max_train_samples 100 \
# --max_eval_samples 10 \
