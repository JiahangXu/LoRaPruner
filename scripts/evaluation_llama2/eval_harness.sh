# python3 main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained=yahma/llama-7b-hf \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
#     --device cuda:0 \
#     --batch_size 1

#!/bin/bash
source /home/aiscuser/anaconda3/bin/activate py39

export PYTHONPATH='.'

cd lm-evaluation-harness
pip install -e .
cd ..

pip install sentencepiece
pip install deepspeed
pip install accelerate
pip install datasets
pip install evaluate
pip install mlflow
pip install torch==2.0.1

echo "check deepspeed"
pip list | grep deepspeed
echo "check torch"
pip list | grep torch

base_model=meta-llama/Llama-2-7b-hf
pretrained_path=$1

python ./lm-evaluation-harness/main.py \
    --model lora-pruner \
    --model_args pretrained=$base_model,peft=$pretrained_path,prompt_mark=$2,lora_param=$3 \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018,race_high \
    --device cuda:0 \
    --output_path results/results.json \
    --no_cache

python ./lm-evaluation-harness/generate.py results/results.json

# nohup bash eval_harness.sh > eval_log.txt 2>&1 &