
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=$((10000 + RANDOM % 50000))

swift sft \
    --model /path/to/your/model \
    --model_type qwen2_audio \
    --dataset '/path/to/dataset.jsonl' \
    --split_dataset_ratio 0.0 \
    --lora_rank 32 \
    --lora_dropout 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --report_to wandb \
    --gradient_accumulation_steps 8 \
    --save_steps 200 \
    --save_total_limit 5 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 1 \
    --max_length 8192 \
    --output_dir './swift_output/outputs_stage2_RSFT' \
    --dataloader_num_workers 16 \
    --train_type lora \
    --target_modules q_proj v_proj 