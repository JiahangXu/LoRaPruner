export CUDA_VISIBLE_DEVICES=4 \
&& bash ./scripts/evaluation_llama2/eval_harness.sh \
&& bash ./scripts/evaluation_llama2/eval_mmlu.sh \
&& bash ./scripts/evaluation_llama2/eval_bbh.sh
