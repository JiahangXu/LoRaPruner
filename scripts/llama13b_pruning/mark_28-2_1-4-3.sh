#!/bin/bash

export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

deepspeed --num_nodes=1 --num_gpus=8 --master_port=16112 train.py \
  --deepspeed ds3_offload.json \
  --pruning_type structured_heads+structured_mlp+hidden \
  --target_sparsity 0.2382 \
  --sparsity_epsilon 0.005 \
  --model_name_or_path decapoda-research/llama-13b-hf \
  --num_train_epochs 8 \
  --learning_rate 5e-5 \
  --reg_learning_rate 0.05 \
  --lagrangian_warmup_epochs 4 \
  --max_seq_length 1024 \
  --task_name llama13b_mark28-2_1-4-3 \
  --do_train \
  --do_eval \
  --sparsity_scheduler cubic \
  --dataset_name alpaca-gpt4 \
  --eval_dataset_name wikitext \
  --train_file /mnt/data/LPM/alpaca_gpt4_data_26k.json \
  --droprate_init 0.01 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --training_objective LM \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR/ \
  --cache_dir /dev/shm \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 40 \
  --gradient_checkpointing=True \
  --logging_first_step \
  --logging_steps 10 \
  --disable_tqdm True \
  --fp16 false \
  --random_init=False  |& tee $OUTPUT_DIR/output.log \
