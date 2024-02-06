lr=$1
mark=${2}_lr${lr}
ckpt_path=$3

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
