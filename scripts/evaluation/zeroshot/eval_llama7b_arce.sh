CUDA_VISIBLE_DEVICE3=0 python evaluation.py \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --pretrained_pruned_model ${1} \
  --dataset_name ai2_arc \
  --dataset_config_name ARC-Easy \
  --cache_dir /dev/shm/ \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --output_dir output \
  --eval_prompt_type ${2} \