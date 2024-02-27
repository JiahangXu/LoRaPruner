source /home/aisilicon/miniconda3/bin/activate compresso
which python

export PYTHONPATH='.'

merged_prune_model=$1
zs_path=$2
peft_path=$3
prompt_mark=$4
peft_mark="${peft_path##*/}"

cp $zs_path/zs.pt $peft_path

python ./lm-evaluation-harness/main.py \
    --model lora-pruner \
    --model_args pretrained=$merged_prune_model,peft=$peft_path,prompt_mark=$prompt_mark,peft_mode=True \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018,race_high \
    --device cuda:0 \
    --output_path results/llama2_${peft_mark}_${prompt_mark}.json \
    --no_cache

python ./lm-evaluation-harness/summary.py results/llama2_${peft_mark}_${prompt_mark}.json

# nohup bash eval_harness.sh > eval_log.txt 2>&1 &