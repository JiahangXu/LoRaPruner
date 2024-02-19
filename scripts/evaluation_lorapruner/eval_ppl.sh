# (compresso)
export PYTHONPATH='.'

lora_ckpt=$1

python ./eval_ppl/eval_ppl.py \
    --max_seq_len 1024 \
    --model_type lora_pruner \
    --base_model meta-llama/Llama-2-7b-hf \
    --lora_ckpt $lora_ckpt \
    --prompt_mark 0
