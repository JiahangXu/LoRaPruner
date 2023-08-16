<h1 align="center"> 
<p> LoRaPruner</p>
</h1>

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

## Training(train.py)

```bash
deepspeed --num_nodes=1 --num_gpus=8 train.py \
  --deepspeed ds3_offload.json \
  --pruning_type structured_heads+structured_mlp+hidden+layer \
  --target_sparsity 0.5 \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --num_train_epochs 10 \
  --learning_rate 5e-5 \
  --reg_learning_rate 0.1 \
  --lagrangian_warmup_epochs 2 \
  --max_seq_length 1024 \
  --do_train \
  --do_eval \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --training_objective LM \
  --overwrite_output_dir \
  --output_dir [OUTPUT_DIR]/ \
  --cache_dir [CACHE_DIR] \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --gradient_checkpointing=True \
  --fp16 false \
  --random_init=False |& tee [OUTPUT_DIR]/output.log \
```
## Finetuning(train.py)

```bash
deepspeed --num_nodes=1 --num_gpus=8 train.py \
  --deepspeed ds3_offload.json \
  --pruning_type None \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --pretrained_pruned_model [Pruned_Model_Path] \
  --num_train_epochs 10 \
  --learning_rate 5e-5 \
  --max_seq_length 1024 \
  --do_train \
  --do_eval \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --training_objective LM \
  --overwrite_output_dir \
  --output_dir [OUTPUT_DIR]/ \
  --cache_dir [CACHE_DIR] \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --gradient_checkpointing=True \
  --fp16 false \
  --random_init=False |& tee [OUTPUT_DIR]/output.log \
```

## Evaluation (evaluation.py)

```bash
deepspeed --num_nodes=1 --num_gpus=8 evaluation.py \
  --deepspeed ds3_offload.json \
  --pruning_type None \
  --model_name_or_path decapoda-research/llama-7b-hf \
  --pretrained_pruned_model [Pruned_Model_Path] \
  --max_seq_length 1024 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --overwrite_output_dir \
  --output_dir [OUTPUT_DIR]/ \
  --cache_dir [CACHE_DIR] \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param Q.V \
  --lora_layers 32 \
  --fp16 false \
  --random_init=False |& tee [OUTPUT_DIR]/output.log \

```

