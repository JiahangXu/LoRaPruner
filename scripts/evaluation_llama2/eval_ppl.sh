# (compresso)
export PYTHONPATH='.'

python ./eval_ppl/eval_ppl.py \
    --max_seq_len 1024 \
    --model_type lora_pruner \
    --base_model $1 \
    --prompt_mark 0
