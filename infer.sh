#python ./llava/eval/run_llava.py \
#    --model-path "/localscratch/gna23/LLaVA/downloads/checkpoint_merged/llava_merged2/" \
#    --image-file "/localscratch/gna23/LLaVA/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"

#python ./llava/eval/run_llava.py \
#    --model-path "liuhaotian/llava-v1.5-7b" \
#    --image-file "/localscratch/gna23/LLaVA/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"\

python ./llava/eval/run_llava.py \
    --model-path "/localscratch/gna23/LLaVA/downloads/checkpoints/llava_lora_fn_weights7" \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --image-file "/localscratch/gna23/LLaVA/dataset/images/be76e1f3-4d58-499c-b7ff-1a0aa24f8260.jpg" \
    --query "How many laptop?"