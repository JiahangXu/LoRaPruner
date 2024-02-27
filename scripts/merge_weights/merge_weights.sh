#!/bin/bash
export PYTHONPATH='.'

pruned_lorazs_path=$1 # mark25
output_dir=$2

python merge_weights.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf  \
  --pretrained_pruned_model $pruned_lorazs_path \
  --training_objective LM \
  --output_dir $output_dir \
  --cache_dir ../cache \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.K.V.O.F \
  --lora_layers 32

echo "MERGE WEIGHTS FINISHED"
