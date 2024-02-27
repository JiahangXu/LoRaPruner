base_model_path=$1
output_dir=$2

source /home/aisilicon/miniconda3/bin/activate compresso
which python
bash ./scripts/merge_weights/merge_weights.sh $base_model_path ./llama_pruned
bash ./scripts/merge_weights/merge_zs.sh ./llama_pruned $base_model_path $output_dir
rm -r ./llama_pruned

source /home/aisilicon/miniconda3/bin/activate llm-pruner
which python
# bash ./scripts/evaluation_llama2/eval_mmlu.sh $output_dir 0
# bash ./scripts/evaluation_llama2/eval_bbh.sh $output_dir 0

# bash ./scripts/evaluation_llama2/eval_mmlu.sh $output_dir 1-1
bash ./scripts/evaluation_llama2/eval_bbh.sh $output_dir 1-1

bash ./scripts/evaluation_llama2/eval_mmlu.sh $output_dir 1
bash ./scripts/evaluation_llama2/eval_bbh.sh $output_dir 1
