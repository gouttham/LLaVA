#python ./llava/eval/run_llava.py \
#    --model-path "/localscratch/gna23/LLaVA/downloads/checkpoint_merged/llava_merged2/" \
#    --image-file "/localscratch/gna23/LLaVA/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"

#python ./llava/eval/run_llava.py \
#    --model-path "liuhaotian/llava-v1.5-7b" \
#    --image-file "/localscratch/gna23/LLaVA/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"\

#python ./llava/eval/run_llava0.py \
#    --model-path "/localscratch/gna23/LLaVA/downloads/checkpoints/llava_lora_fn_cls2_4cls" \
#    --model-base "liuhaotian/llava-v1.5-7b" \
#    --image-file "/localscratch/gna23/LLaVA/v2/cd_images/7fada345-8e1e-4956-a1bb-79735da8928f_no.jpg" \
#    --query "Are given two images are similar ?"

python ./llava/eval/run_llava0.py \
    --model-path "/localscratch/gna23/LLaVA/downloads/checkpoints/llava_lora_fn_cls2_4cls_copy" \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --image-file "/localscratch/gna23/LLaVA/v2/cd_images/7fada345-8e1e-4956-a1bb-79735da8928f_no.jpg" \
    --query "Are given two images are similar ?"
