# (compresso)
export PYTHONPATH='.'

base_model=$1
peft_ckpt=$2

python ./eval_ppl/eval_ppl.py \
    --max_seq_len 1024 \
    --model_type llm_pruner \
    --ckpt $base_model \
    --lora_ckpt $peft_ckpt \
    --prompt_mark 0
