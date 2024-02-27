source /home/aisilicon/miniconda3/bin/activate llm-pruner
export PYTHONPATH='.'

python ./lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$1 \
    --tasks openbookqa \
    --device cuda:0 \
    --no_cache
