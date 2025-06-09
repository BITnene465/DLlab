$env:CUDA_VISIBLE_DEVICES=0
$SPLIT = "mmbench_dev_20230712"

cd ../LLaVA
python -m llava.eval.model_vqa_mmbench `
    --model-path liuhaotian/llava-v1.5-7b `
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv `
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b.jsonl `
    --single-pred-prompt `
    --temperature 0 `

New-Item -Path "playground/data/eval/mmbench/answers_upload/$SPLIT" -ItemType Directory -Force

python scripts/convert_mmbench_for_submission.py `
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv `
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT `
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT `
    --experiment llava-v1.5-7b