export PYTHONPATH='.'

base_model=$1 # e.g., meta-llama/Llama-2-7b-hf
tune_ckpt_name=$2 
prune_ckpt=$3
epoch=$4
tune_id="${tune_ckpt_name##*/}"

# cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
# mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

python ./lm-evaluation-harness/main.py \
    --model llm-pruner \
    --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name,config_pretrained=$base_model \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018,race_high \
    --device cuda:0 \
    --output_path results/${tune_id}_$epoch.json \
    --no_cache

python ./lm-evaluation-harness/summary.py results/${tune_id}_$epoch.json
