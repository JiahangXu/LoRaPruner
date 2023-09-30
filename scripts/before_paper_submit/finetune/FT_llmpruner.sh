lr=$1
mark=${2}_lr${lr}
ckpt_path=$3

# deepspeed --num_nodes=1 --num_gpus=1 --master_port=16112 merge_weights.py \
#   --pruning_type None \
#   --target_sparsity 0. \
#   --sparsity_epsilon 0.005 \
#   --model_name_or_path decapoda-research/llama-7b-hf \
#   --pretrained_pruned_model $ckpt_path \
#   --task_name None \
#   --training_objective LM \
#   --overwrite_output_dir \
#   --output_dir ./ \
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
#   --random_init=False |& tee $OUTPUT_DIR/output.log \

CUDA_VISIBLE_DEVICES=0 python post_training_lora_pruner.py \
    --prune_model ./llama_pruned \
    --pretrained_pruned_model $ckpt_path \
    --output_dir finetune_results/${mark}_bs8_AlpacaGPT4 \
    --wandb_project vLoRaPruner \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate $lr \
    --batch_size 8 \
    --micro_batch_size 1
