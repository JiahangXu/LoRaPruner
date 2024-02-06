#!/bin/bash

cd /home/jiahangxu/LoRaPruner

export PYTHONPATH='.'

export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true


base_path=$1
pruned_lorazs_path=$2 # mark25
prompt_mark=$3
cuda=$4
output_dir="${base_path##*/}"

echo $output_dir

# python merge_zs.py \
#   --pruning_type None \
#   --target_sparsity 0.3 \
#   --sparsity_epsilon 0.005 \
#   --model_name_or_path $base_path \
#   --num_train_epochs 1 \
#   --learning_rate 1e-6 \
#   --reg_learning_rate 0.05 \
#   --lagrangian_warmup_epochs 0 \
#   --pretrained_pruned_model $pruned_lorazs_path \
#   --max_seq_length 1024 \
#   --task_name debug \
#   --do_train \
#   --do_eval \
#   --dataset_name alpaca-gpt4 \
#   --eval_dataset_name wikitext \
#   --train_file /mnt/data/LPM/alpaca_gpt4_data.json \
#   --droprate_init 0.01 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --training_objective LM \
#   --overwrite_output_dir \
#   --output_dir $output_dir/ \
#   --cache_dir /dev/shm/ \
#   --use_lora True \
#   --lora_rank 8 \
#   --lora_train_bias none \
#   --lora_alpha 8.0 \
#   --lora_param Q.V \
#   --lora_layers 32 \
#   --gradient_checkpointing=True \
#   --logging_first_step \
#   --logging_steps 10 \
#   --disable_tqdm True \
#   --fp16 false \
#   --random_init=False \
#   --lr_scheduler_type cosine  |& tee $OUTPUT_DIR/output.log \


# python ./eval_ppl/eval_ppl.py --max_seq_len 1024 --model_type lora_pruner --base_model $output_dir --prompt_mark 0
# python ./eval_ppl/eval_ppl.py --max_seq_len 1024 --model_type lora_pruner --base_model /home/jiahangxu/llama/pruned_llama_1ep_1e-5 --prompt_mark 0 --lora_ckpt ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6 --lora_merged

cd ~/instruct-eval
CUDA_VISIBLE_DEVICES=$cuda python main.py mmlu --model_name llama --model_path /home/jiahangxu/LoRaPruner/$output_dir --prompt_mark $prompt_mark
CUDA_VISIBLE_DEVICES=$cuda python main.py bbh --model_name llama --model_path /home/jiahangxu/LoRaPruner/$output_dir --prompt_mark $prompt_mark


# nohup bash eval_harness.sh > eval_log.txt 2>&1 &