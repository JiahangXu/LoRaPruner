export PYTHONPATH='.'

base_model=$1 # e.g., meta-llama/Llama-2-7b-hf
tune_ckpt_name=$2 
prune_ckpt=$3
tune_id="${tune_ckpt_name##*/}"

python ./instruct-eval/main.py mmlu \
    --model_name llmpruner \
    --model_path  $prune_ckpt/pytorch_model.bin \
    --tokenizer $base_model \
    --lora_path $tune_ckpt_name \
    --prompt_mark 0
