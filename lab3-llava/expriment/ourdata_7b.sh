# /bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./ourdata/questions.jsonl \
    --image-folder ./ourdata/images \
    --answers-file ./ourdata_results/results.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1