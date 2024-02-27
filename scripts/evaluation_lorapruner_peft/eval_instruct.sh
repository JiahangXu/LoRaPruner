merged_prune_model=$1
peft_zs_path=$2
output_dir="${peft_zs_path##*/}"

source /home/aisilicon/miniconda3/bin/activate compresso
which python
bash ./scripts/merge_weights/merge_peft.sh $merged_prune_model $peft_zs_path $output_dir
bash ./scripts/merge_weights/merge_zs.sh $output_dir $peft_zs_path $output_dir

source /home/aisilicon/miniconda3/bin/activate llm-pruner
which python
bash ./scripts/evaluation_llama2/eval_instruct.sh $output_dir
