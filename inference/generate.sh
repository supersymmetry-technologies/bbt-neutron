output_model=/root/shared/beegfs/ldh/test/binbbt/output
export PYTHONPATH="/root/shared/beegfs/ldh/test/binbbt:${PYTHONPATH}"
export DS_ENV_FILE="/root/shared/beegfs/ldh/test/binbbt/config/.deepspeed_env"
export TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX"
export CUTLASS_PATH="/opt/cutlass"
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

# deepspeed --include localhost:0,1,2,3,4,5,6,7 \
#     --master_port 5000 \
# deepspeed --hostfile /root/binbbt/config/hostfile \
#     --master_port 5000 \
#     --include 10.0.1.7:0,1,2,3,4,5,6,7@10.0.1.8:0,1,2,3,4,5,6,7 \
#     --ssh_port 22242 \
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    --master_port 22001 \
    generate.py \
    --task_type byte \
    --validation_dir /root/shared/beegfs/ldh/test/binbbt/data/slim/validation \
    --data_dir /root/shared/beegfs/ldh/test/binbbt/data/slim/byte_arrow_4096_1 \
    --model_name_or_path /root/shared/beegfs/ldh/test/binbbt/output \
    --streaming false \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_eval \
    --max_eval_samples 100 \
    --output_dir "${output_model}" \
    --preprocessing_num_workers 128 \
    --logging_dir "${output_model}/logs" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --log_level "info" \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 4096 \
    --step_size 4096 \
    --patch_size 1 \
    --run_name "${output_model}" \
    --bf16 \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a "${output_model}/generate.log"
    # > "${output_model}/generate.log" 2>&1

# --number_classes 2 \
# --resume_from_checkpoint ${output_model}/checkpoint-20400 \
# --report_to wandb \
# --use_flash_attn \
# --use_peft_lora \
# --max_eval_samples 10 \