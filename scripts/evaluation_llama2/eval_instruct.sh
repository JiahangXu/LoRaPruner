model_path=$1

bash ./scripts/evaluation_llama2/eval_mmlu.sh $model_path 0
bash ./scripts/evaluation_llama2/eval_bbh.sh $model_path 0

bash ./scripts/evaluation_llama2/eval_mmlu.sh $model_path 1-1
bash ./scripts/evaluation_llama2/eval_bbh.sh $model_path 1-1

bash ./scripts/evaluation_llama2/eval_mmlu.sh $model_path 1
bash ./scripts/evaluation_llama2/eval_bbh.sh $model_path 1
