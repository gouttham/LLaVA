python scripts/merge_lora_weights.py \
    --model-path "checkpoints/llama-2-7b-chat-task-qlora" \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --save-model-path "checkpoints/merged/"