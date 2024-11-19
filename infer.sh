#python ./llava/eval/run_llava.py \
#    --model-path "/localscratch/gna23/LLaVA/downloads/checkpoint_merged/llava_merged2/" \
#    --image-file "/localscratch/gna23/LLaVA/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"

#python ./llava/eval/run_llava.py \
#    --model-path "liuhaotian/llava-v1.5-7b" \
#    --image-file "/localscratch/gna23/LLaVA/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"\

python ./llava/eval/run_llava.py \
    --model-path "/localscratch/gna23/LLaVA/downloads/checkpoints/llava_fn_weights3" \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --image-file "/localscratch/gna23/LLaVA/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
    --query "why was this photo taken?"