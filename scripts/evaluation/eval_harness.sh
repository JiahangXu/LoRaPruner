# python3 main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained=yahma/llama-7b-hf \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
#     --device cuda:0 \
#     --batch_size 1

#!/bin/bash
export PYTHONPATH='.'

base_model=decapoda-research/llama-7b-hf
pretrained_path=$1

python ./lm-evaluation-harness/main.py \
    --model lora-pruner \
    --model_args pretrained=$base_model,peft=$pretrained_path \
    --tasks piqa \
    --device cuda:0 \
    --output_path results/results.json \
    --no_cache

python ./eval_ppl/eval_ppl.py \
    --max_seq_len 1024 \
    --model_type lora_pruner \
    --lora_ckpt $pretrained_path

python ./lm-evaluation-harness/generate.py results/results.json

# nohup bash eval_harness.sh > eval_log.txt 2>&1 &  