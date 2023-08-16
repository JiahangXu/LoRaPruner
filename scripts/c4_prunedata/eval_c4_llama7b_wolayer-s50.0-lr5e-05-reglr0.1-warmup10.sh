#!/bin/bash

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

deepspeed --num_nodes=1 --num_gpus=2 train.py \
  --deepspeed ds3_offload.json \
  --pruning_type structured_heads+structured_mlp+hidden+layer \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --pretrained_pruned_model /mnt/data/LoRaPruner/c4_llama7b_wolayer-s50.0-lr5e-05-reglr0.1-warmup10/2023-6-22-16-16/epoch15/best \
  --do_eval \
  --max_seq_length 1024 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --training_objective LM \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --overwrite_output_dir \
  --output_dir output/ \
  --cache_dir /dev/shm/gs/ \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --fp16 false \
  --random_init=False |& tee output/output.log \





