export PYTHONPATH='.'

export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

baseline_pruned_model=$1
pretrained_path=$2

deepspeed --num_nodes=1 --num_gpus=1 --master_port=16112 merge_weights.py \
  --pruning_type None \
  --target_sparsity 0. \
  --sparsity_epsilon 0.005 \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --pretrained_pruned_model $baseline_pruned_model \
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
  --random_init=False |& tee $OUTPUT_DIR/output.log

cp $baseline_pruned_model/zs.pt $pretrained_path/
cp $baseline_pruned_model/l0_module.pt $pretrained_path/


python ./eval_ppl/eval_ppl.py \
    --max_seq_len 1024 \
    --model_type lora_pruner \
    --base_model ./llama_pruned \
    --lora_ckpt $pretrained_path \
    --prompt_mark $3
