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

base_model=~/working/myfastnn/LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-32/epoch4
pretrained_path=$1

cp -r $base_model/llama_pruned ./
cp $base_model/zs.pt $pretrained_path/
cp $base_model/l0_module.pt $pretrained_path/


python ./lm-evaluation-harness/main.py \
    --model lora-pruner \
    --model_args pretrained=./llama_pruned,peft=$pretrained_path \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018 \
    --device cuda:0 \
    --output_path results/results.json \
    --no_cache

python ./lm-evaluation-harness/generate.py results/results.json

# nohup bash eval_harness.sh > eval_log.txt 2>&1 &