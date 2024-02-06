export PYTHONPATH='.'

base_model=./llama_pruned
pretrained_path=$1

python ./lm-evaluation-harness/main.py \
    --model lora-pruner \
    --model_args pretrained=$base_model,peft=$pretrained_path,prompt_mark=$3,peft_mode=True \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018 \
    --device cuda:0 \
    --output_path results/results.json \
    --no_cache

python ./lm-evaluation-harness/generate.py results/results.json

# nohup bash eval_harness.sh > eval_log.txt 2>&1 &