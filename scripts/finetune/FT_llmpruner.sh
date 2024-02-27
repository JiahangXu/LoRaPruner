echo $CUDA_VISIBLE_DEVICES

base_model_path=${1}
lr=$2
mark=${3}_lr${lr}
ckpt_path=$4

source /home/aisilicon/miniconda3/bin/activate compresso
which python
python post_training_lora_pruner.py \
    --prune_model $base_model_path \
    --pretrained_pruned_model $ckpt_path \
    --output_dir finetune_results/${mark}_bs8_AlpacaGPT4 \
    --wandb_project vLoRaPruner \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate $lr \
    --batch_size 8 \
    --micro_batch_size 1

bash ./scripts/evaluation_lorapruner_peft/eval_all.sh $base_model_path $ckpt_path finetune_results/${mark}_bs8_AlpacaGPT4
bash ./scripts/evaluation_lorapruner_peft/eval_instruct.sh $base_model_path finetune_results/${mark}_bs8_AlpacaGPT4
