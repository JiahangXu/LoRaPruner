# export CUDA_VISIBLE_DEVICES=7 \
# && bash scripts/evaluation_llmpruner/eval_harness.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5.4b_mix ../LLM-Pruner/prune_log/llama2_prune_5.4b_mix final \
# && bash scripts/evaluation_llmpruner/eval_harness.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5.4b_mix_gpt4alpaca ../LLM-Pruner/prune_log/llama2_prune_5.4b_mix final \
# && bash scripts/evaluation_llmpruner/eval_harness.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5b_mix ../LLM-Pruner/prune_log/llama2_prune_5b_mix final \
# && bash scripts/evaluation_llmpruner/eval_harness.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5b_mix_gpt4alpaca ../LLM-Pruner/prune_log/llama2_prune_5b_mix final \
# && bash scripts/evaluation_llmpruner/eval_harness.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_4.5b_mix ../LLM-Pruner/prune_log/llama2_prune_4.5b_mix final \
# && bash scripts/evaluation_llmpruner/eval_harness.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_4.5b_mix_gpt4alpaca ../LLM-Pruner/prune_log/llama2_prune_4.5b_mix final \

export CUDA_VISIBLE_DEVICES=1 \
&& bash scripts/evaluation_llmpruner/eval_mmlu.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5.4b_mix ../LLM-Pruner/prune_log/llama2_prune_5.4b_mix \
&& bash scripts/evaluation_llmpruner/eval_bbh.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5.4b_mix ../LLM-Pruner/prune_log/llama2_prune_5.4b_mix \
&& bash scripts/evaluation_llmpruner/eval_mmlu.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5.4b_mix_gpt4alpaca ../LLM-Pruner/prune_log/llama2_prune_5.4b_mix \
&& bash scripts/evaluation_llmpruner/eval_bbh.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5.4b_mix_gpt4alpaca ../LLM-Pruner/prune_log/llama2_prune_5.4b_mix

export CUDA_VISIBLE_DEVICES=2 \
&& bash scripts/evaluation_llmpruner/eval_mmlu.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5b_mix ../LLM-Pruner/prune_log/llama2_prune_5b_mix \
&& bash scripts/evaluation_llmpruner/eval_bbh.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5b_mix ../LLM-Pruner/prune_log/llama2_prune_5b_mix \
&& bash scripts/evaluation_llmpruner/eval_mmlu.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5b_mix_gpt4alpaca ../LLM-Pruner/prune_log/llama2_prune_5b_mix \
&& bash scripts/evaluation_llmpruner/eval_bbh.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_5b_mix_gpt4alpaca ../LLM-Pruner/prune_log/llama2_prune_5b_mix

export CUDA_VISIBLE_DEVICES=3 \
&& bash scripts/evaluation_llmpruner/eval_mmlu.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_4.5b_mix ../LLM-Pruner/prune_log/llama2_prune_4.5b_mix \
&& bash scripts/evaluation_llmpruner/eval_bbh.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_4.5b_mix ../LLM-Pruner/prune_log/llama2_prune_4.5b_mix \
&& bash scripts/evaluation_llmpruner/eval_mmlu.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_4.5b_mix_gpt4alpaca ../LLM-Pruner/prune_log/llama2_prune_4.5b_mix \
&& bash scripts/evaluation_llmpruner/eval_bbh.sh meta-llama/Llama-2-7b-hf ../LLM-Pruner/tune_log/llama2_4.5b_mix_gpt4alpaca ../LLM-Pruner/prune_log/llama2_prune_4.5b_mix

# nohup bash scripts/evaluation_llmpruner/eval_all.sh > eval_llmpruner_log.txt 2>&1 &