#!/bin/bash
source /home/aiscuser/anaconda3/bin/activate py39

export PYTHONPATH='.'

cd lm-evaluation-harness
pip install -e .
cd ..
pip install peft

params_path=$1 # 4.8b
base_model=decapoda-research/llama-13b-hf # e.g., decapoda-research/llama-13b-hf
tune_ckpt_name=/mnt/data/LoRaPruner/LLM-Pruner-Baseline/llama7b_${params_path}_mix/tune_log
prune_ckpt=/mnt/data/LoRaPruner/LLM-Pruner-Baseline/llama7b_${params_path}_mix/prune_log
epoch=1400
tune_id="${tune_ckpt_name##*/}"

cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

# python main.py \
#     --model llm-pruner \
#     --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018 \
#     --device cuda:0 \
#     --output_path results/${tune_id}_$epoch.json \
#     --no_cache

python ./lm-evaluation-harness/main.py \
    --model llm-pruner \
    --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model \
    --tasks nq_open \
    --device cuda:0 \
    --output_path results/${tune_id}_$epoch.json \
    --no_cache
