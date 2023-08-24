baseline_pruned_model=/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-23/epoch4
pretrained_pruned_model=${1}
cp $baseline_pruned_model/zs.pt $pretrained_pruned_model/
cp $baseline_pruned_model/l0_module.pt $pretrained_pruned_model/

cp -r $baseline_pruned_model/llama_pruned ./


deepspeed --num_nodes=1 --num_gpus=1 train.py \
  --deepspeed ds3_offload.json \
  --pruning_type structured_heads+structured_mlp+hidden+mlp_layer+head_layer \
  --model_name_or_path ./llama_pruned \
  --pretrained_pruned_model $pretrained_pruned_model \
  --do_eval \
  --max_seq_length 1024 \
  --eval_dataset_name wikitext \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --training_objective LM \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --overwrite_output_dir \
  --output_dir output/ \
  --cache_dir /dev/shm/ \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --fp16 false \
  --random_init=False \
  --prompt_mark ${2}
