CUDA_VISIBLE_DEVICE3=0 python evaluation.py \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --pretrained_pruned_model /mnt/data/LoRaPruner/math_llama7b_wolayer_research-s30.0-lr5e-05-reglr0.1-warmup10/2023-6-21-6-14/epoch18 \
  --dataset_name storycloze \
  --task storycloze \
  --train_file /mnt/data/LPM/storycloze \
  --cache_dir /dev/shm/gs \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --output_dir output \