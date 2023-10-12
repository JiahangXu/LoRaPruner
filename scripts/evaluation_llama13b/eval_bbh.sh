#!/bin/bash
export PYTHONPATH='.'


export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

pruned_lorazs_path=$1 # mark25

deepspeed --num_nodes=1 --num_gpus=1 --master_port=16112 merge_weights.py \
  --pruning_type None \
  --target_sparsity 0. \
  --sparsity_epsilon 0.005 \
  --model_name_or_path decapoda-research/llama-13b-hf \
  --pretrained_pruned_model $pruned_lorazs_path \
  --task_name None \
  --training_objective LM \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR/ \
  --cache_dir /dev/shm/ \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param $3 \
  --lora_layers 32 \
  --gradient_checkpointing=True \
  --logging_first_step \
  --logging_steps 10 \
  --disable_tqdm True \
  --fp16 false \
  --random_init=False |& tee $OUTPUT_DIR/output.log \

echo "STEP 1 FINISHED"
python ./eval_ppl/eval_ppl.py --max_seq_len 1024 --model_type lora_pruner --base_model ./llama_pruned --prompt_mark 0


python merge_zs.py \
  --pruning_type None \
  --target_sparsity 0.3 \
  --sparsity_epsilon 0.005 \
  --model_name_or_path ./llama_pruned \
  --num_train_epochs 1 \
  --learning_rate 1e-6 \
  --reg_learning_rate 0.05 \
  --lagrangian_warmup_epochs 0 \
  --pretrained_pruned_model $pruned_lorazs_path \
  --max_seq_length 1024 \
  --task_name debug \
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
  --cache_dir /dev/shm/ \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param $3 \
  --lora_layers 32 \
  --gradient_checkpointing=True \
  --logging_first_step \
  --logging_steps 10 \
  --disable_tqdm True \
  --fp16 false \
  --random_init=False \
  --lr_scheduler_type cosine  |& tee $OUTPUT_DIR/output.log \

echo "STEP 2 FINISHED"

python ./eval_ppl/eval_ppl.py --max_seq_len 1024 --model_type lora_pruner --base_model ./llama_pruned --prompt_mark 0

source /home/aiscuser/anaconda3/bin/activate instruct-eval

cd instruct-eval
python main.py bbh --model_name llama --model_path ../llama_pruned --prompt_mark $2
