# deepspeed --num_nodes=1 --num_gpus=1 train.py \
#   --deepspeed ds3_offload.json \
#   --pruning_type structured_heads+structured_mlp+hidden+mlp_layer+head_layer \
#   --model_name_or_path decapoda-research/llama-7b-hf \
#   --pretrained_pruned_model ${1} \
#   --do_eval \
#   --max_seq_length 1024 \
#   --eval_dataset_name wikitext2_eval \
#   --dataset_name wikitext2_eval \
#   --dataset_config_name wikitext-2-raw-v1 \
#   --training_objective LM \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --overwrite_output_dir \
#   --output_dir output/ \
#   --cache_dir /dev/shm/ \
#   --use_lora True \
#   --lora_rank 8 \
#   --lora_train_bias none \
#   --lora_alpha 8.0 \
#   --lora_param Q.V \
#   --lora_layers 32 \
#   --fp16 false \
#   --random_init=False \
#   --prompt_mark ${2}

export PYTHONPATH='.'

python ./eval_ppl/eval_ppl.py \
    --max_seq_len 1024 \
    --model_type lora_pruner \
    --eval_c4 \
    --lora_ckpt ${1} \
    --prompt_mark ${2}