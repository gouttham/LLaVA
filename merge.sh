#python scripts/merge_lora_weights.py \
#    --model-path "/localscratch/gna23/LLaVA/downloads/checkpoints/llava_lora_weights" \
#    --model-base "liuhaotian/llava-v1.5-7b" \
#    --save-model-path "/localscratch/gna23/LLaVA/downloads/checkpoint_merged/llava_merged/"


python scripts/merge_lora_weights.py \
    --model-path "/localscratch/gna23/LLaVA/downloads/checkpoints/llava_lora_weights3" \
    --model-base "/localscratch/gna23/LLaVA/downloads/liuhaotian/llava-v1.5-7b" \
    --save-model-path "/localscratch/gna23/LLaVA/downloads/checkpoint_merged/llava_merged3/"