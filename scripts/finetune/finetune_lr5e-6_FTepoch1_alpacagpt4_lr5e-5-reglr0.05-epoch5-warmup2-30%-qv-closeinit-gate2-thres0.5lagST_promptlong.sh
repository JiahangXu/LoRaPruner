#!/bin/bash

export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

# output directory
pruning_type=None

# baseline_pruned_model=/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-23/epoch4
baseline_pruned_model=/mnt/data/LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-32/epoch4
cp -r $baseline_pruned_model/llama_pruned/ ./

deepspeed --num_nodes=1 --num_gpus=8 --master_port=16112 train.py \
  --deepspeed ds3_offload.json \
  --pruning_type None \
  --target_sparsity 0.3 \
  --sparsity_epsilon 0.005 \
  --model_name_or_path ./llama_pruned \
  --num_train_epochs 1 \
  --learning_rate 5e-6 \
  --reg_learning_rate 0.05 \
  --lagrangian_warmup_epochs 0 \
  --pretrained_pruned_model $baseline_pruned_model \
  --max_seq_length 1024 \
  --task_name gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST \
  --do_train \
  --do_eval \
  --dataset_name alpaca-gpt4 \
  --eval_dataset_name wikitext \
  --train_file /mnt/data/LPM/alpaca_gpt4_data.json \
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
