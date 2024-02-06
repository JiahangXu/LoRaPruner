#!/bin/bash
export PYTHONPATH='.'

export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

pruned_lorazs_path=$1 # mark25
finetune_lora_path=$2 # mark25FT9 / mark25FT15

deepspeed --num_nodes=1 --num_gpus=1 --master_port=16112 merge_weights.py \
  --pruning_type None \
  --target_sparsity 0. \
  --sparsity_epsilon 0.005 \
  --model_name_or_path meta-llama/Llama-2-7b-hf  \
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
  --lora_param Q.K.V.O.F \
  --lora_layers 32 \
  --gradient_checkpointing=True \
  --logging_first_step \
  --logging_steps 10 \
  --disable_tqdm True \
  --fp16 false \
  --random_init=False |& tee $OUTPUT_DIR/output.log \

echo "STEP 1 FINISHED"
python ./eval_ppl/eval_ppl.py --max_seq_len 1024 --model_type lora_pruner --base_model ./merged_llama_mark25 --prompt_mark 0

deepspeed --num_nodes=1 --num_gpus=1 --master_port=16112 merge_weights.py \
  --pruning_type None \
  --target_sparsity 0. \
  --sparsity_epsilon 0.005 \
  --model_name_or_path ./llama_mergezs \
  --pretrained_pruned_model $finetune_lora_path \
  --task_name None \
  --training_objective LM \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR/ \
  --cache_dir /dev/shm/ \
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
  --random_init=False |& tee $OUTPUT_DIR/output.log \

echo "STEP 2 FINISHED"
python ./eval_ppl/eval_ppl.py --max_seq_len 1024 --model_type lora_pruner --base_model ./merged_llama_mark25 --prompt_mark 0

python merge_zs.py \
  --pruning_type None \
  --target_sparsity 0.3 \
  --sparsity_epsilon 0.005 \
  --model_name_or_path ./llama_mergezs \
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
  --lora_param Q.V \
  --lora_layers 32 \
  --gradient_checkpointing=True \
  --logging_first_step \
  --logging_steps 10 \
  --disable_tqdm True \
  --fp16 false \
  --random_init=False \
  --lr_scheduler_type cosine  |& tee $OUTPUT_DIR/output.log \

echo "STEP 3 FINISHED"

python ./eval_ppl/eval_ppl.py --max_seq_len 1024 --model_type lora_pruner --base_model ./merged_llama_mark25 --prompt_mark 0
# cp $baseline_pruned_model/zs.pt $pretrained_path/
# cp $baseline_pruned_model/l0_module.pt $pretrained_path/

# # mark28-2_1.6e-3-epoch1
# # bash /home/jiahangxu/working/LoRaPruner/scripts/evaluation_finetune/merge_weights_zs.sh ./baseline_pruned_model_mark28-2 ~/working/myfastnn/LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-13-3-48/epoch1/

# # mark25FT9
# # bash /home/jiahangxu/working/LoRaPruner/scripts/evaluation_finetune/merge_weights_zs.sh ~/working/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6 ~/working/myfastnn/LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_5e-5-s30.0-lr5e-05-reglr0.05-warmup0/2023-9-3-4-2/epoch0

# # mark25FT15
# # bash /home/jiahangxu/working/LoRaPruner/scripts/evaluation_finetune/merge_weights_zs.sh ~/working/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6 ~/working/myfastnn/LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_2e-4-s30.0-lr0.0002-reglr0.05-warmup0/2023-9-4-10-39/epoch0
  
# # mark28-5_4e-4-epoch1
# # bash /home/jiahangxu/working/LoRaPruner/scripts/evaluation_finetune/merge_weights_zs.sh ~/working/myfastnn/LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6 ~/working/myfastnn/LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-13-17-56/epoch1

# # nohup bash eval_harness.sh > eval_log.txt 2>&1 &