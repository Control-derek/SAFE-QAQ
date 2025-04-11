# add think_length_reward_300 in --reward_funcs for the model without sft
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model /path/to/your/model \
    --model_type qwen2_audio \
    --external_plugins ./plugin.py \
    --reward_funcs new_adaptive_classification prompt_format stage_classification \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --train_type lora \
    --target_modules q_proj v_proj \
    --lora_rank 32 \
    --torch_dtype bfloat16 \
    --dataset '/path/to/dataset.jsonl' \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 3 \
    --num_generations 9 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 8 \
    --split_dataset_ratio 0.0 \
    --save_steps 100 \
    --save_total_limit 12 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 1 \
    --max_length 8192 \
    --output_dir './swift_output/outputs_stage3' \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --temperature 0.9 \
    --deepspeed zero2 \
    --log_completions true \
    --report_to wandb