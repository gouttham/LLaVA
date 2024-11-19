deepspeed /localscratch/gna23/LLaVA/llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --bits 4 \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version llava_llama_2 \
    --data_path /localscratch/gna23/LLaVA/dataset/train/dataset.json \
    --image_folder /localscratch/gna23/LLaVA/dataset/images/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /localscratch/gna23/LLaVA/llava/downloads/checkpoints/llava_lora_weights3 \
    --num_train_epochs 1 \
    --per_devicetrain_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --cache_dir /localscratch/gna23/LLaVA/downloads/ \
    --tune_mm_mlp_adapter True
    --low_cpu_mem_usage False