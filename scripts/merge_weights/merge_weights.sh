#!/bin/bash
export PYTHONPATH='.'

export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

pruned_lorazs_path=$1 # mark25
finetune_lora_path=$2 # mark25FT9 / mark25FT15

python merge_weights.py \
  --pruning_type None \
  --target_sparsity 0. \
  --sparsity_epsilon 0.005 \
  --model_name_or_path meta-llama/Llama-2-7b-hf  \
  --pretrained_pruned_model $pruned_lorazs_path \
  --task_name None \
  --training_objective LM \
  --overwrite_output_dir \
  --output_dir ./ \
  --cache_dir ../cache \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.K.V.O.F \
  --lora_layers 32 \
  --gradient_checkpointing=True \
  --logging_first_step \
  --logging_steps 10 \
  --disable_tqdm True \
  --fp16 false \
  --random_init=False

echo "STEP 1 FINISHED"
