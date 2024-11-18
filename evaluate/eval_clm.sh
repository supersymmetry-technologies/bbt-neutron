# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
export HF_ENDPOINT="https://hf-mirror.com"

accelerate launch \
    --config_file /root/shared/beegfs/ldh/test/binbbt/config/accelerate_config.yaml \
    --main_process_port 22001 \
    -m lm_eval \
    --model hf \
    --model_args pretrained=/root/shared/beegfs/ldh/test/binbbt/output \
    --tasks leaderboard,mmlu,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,truthfulqa,gsm8k \
    --output_path /root/shared/beegfs/ldh/test/binbbt/output/lm_eval_result \
    --batch_size auto:8 \
    --trust_remote_code

# lm_eval \
#     --model hf \
#     --model_args pretrained=/root/shared/beegfs/ldh/llama/output/checkpoint-46000 \
#     --tasks leaderboard,mmlu,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,truthfulqa,gsm8k \
#     --output_path /root/shared/beegfs/ldh/llama/output/checkpoint-46000/lm_eval_result \
#     --batch_size auto:4 \
#     --trust_remote_code

# --log_samples \
# --num_fewshot 3 \
