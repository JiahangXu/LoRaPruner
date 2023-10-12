#!/bin/bash
source /home/aiscuser/anaconda3/bin/activate py39

export PYTHONPATH='.'

cd lm-evaluation-harness
pip install -e .
cd ..

export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

base_model=decapoda-research/llama-7b-hf
pretrained_path=$1



python ./lm-evaluation-harness/main.py \
    --model lora-pruner \
    --model_args pretrained=$pretrained_path,prompt_mark=$2 \
    --tasks race_high \
    --device cuda:0 \
    --output_path results/results.json \
    --no_cache

# python ./lm-evaluation-harness/generate.py results/results.json

# nohup bash eval_harness.sh > eval_log.txt 2>&1 &