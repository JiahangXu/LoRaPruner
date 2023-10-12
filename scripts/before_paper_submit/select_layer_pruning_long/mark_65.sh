#!/bin/bash

export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

deepspeed --num_nodes=1 --num_gpus=8 --master_port=16112 train.py \
  --deepspeed ds3_offload.json \
  --pruning_type structured_heads+structured_mlp+hidden \
  --target_sparsity 0.2 \
  --sparsity_epsilon 0.005 \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --num_train_epochs 8 \
  --prepruning_finetune_epochs 1 \
  --lagrangian_warmup_epochs 5 \
  --learning_rate 3e-5 \
  --reg_learning_rate 0.05 \
  --max_seq_length 1024 \
  --block_layer_start 4 \
  --block_layer_end 30 \
  --task_name gpt4alpaca-5k_llama7b_promptlong_mark65 \
  --do_train \
  --do_eval \
  --dataset_name alpaca-gpt4 \
  --eval_dataset_name wikitext \
  --train_file /mnt/data/LPM/alpaca_gpt4_data_5k.json \
  --droprate_init 0.01 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --training_objective LM \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR/ \
  --cache_dir /dev/shm/gs \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --gradient_checkpointing=True \
  --logging_first_step \
  --logging_steps 10 \
  --disable_tqdm True \
  --fp16 false \
  --random_init=False  |& tee $OUTPUT_DIR/output.log \
