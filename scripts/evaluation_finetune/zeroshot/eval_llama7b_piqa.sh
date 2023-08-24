baseline_pruned_model=/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-23/epoch4
pretrained_pruned_model=${1}
cp $baseline_pruned_model/zs.pt $pretrained_pruned_model/
cp $baseline_pruned_model/l0_module.pt $pretrained_pruned_model/

cp -r $baseline_pruned_model/llama_pruned ./

CUDA_VISIBLE_DEVICE3=0 python evaluation.py \
  --model_name_or_path ./llama_pruned \
  --pretrained_pruned_model $pretrained_pruned_model \
  --dataset_name piqa \
  --task piqa \
  --cache_dir /dev/shm/ \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --output_dir output \
  --eval_prompt_type ${2} \