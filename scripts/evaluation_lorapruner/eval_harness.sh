source /home/aisilicon/miniconda3/bin/activate compresso
which python

export PYTHONPATH='.'

base_model=meta-llama/Llama-2-7b-hf
pruned_path=$1
result_mark=$2
prompt_mark=$3

python ./lm-evaluation-harness/main.py \
    --model lora-pruner \
    --model_args pretrained=$base_model,peft=$pruned_path,prompt_mark=$prompt_mark \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018,race_high \
    --device cuda:0 \
    --output_path results/llama2_${result_mark}_${prompt_mark}.json \
    --no_cache

python ./lm-evaluation-harness/summary.py results/llama2_${result_mark}_${prompt_mark}.json
