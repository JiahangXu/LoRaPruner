# meta-llama/Llama-2-7b-hf
model_path=$1
prompt_mark=$2

python ./instruct-eval/main.py mmlu \
    --model_name llama \
    --model_path $model_path \
    --prompt_mark $prompt_mark
