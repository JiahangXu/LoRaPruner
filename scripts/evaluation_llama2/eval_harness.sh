export PYTHONPATH='.'

python ./lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,storycloze_2018,race_high \
    --device cuda:0 \
    --output_path results/llama2.json \
    --no_cache

python ./lm-evaluation-harness/generate.py results/llama2.json

