merged_prune_model=$1
zs_path=$2
peft_path=$3

bash ./scripts/evaluation_lorapruner_peft/eval_harness.sh $merged_prune_model $zs_path $peft_path 0
bash ./scripts/evaluation_lorapruner_peft/eval_harness.sh $merged_prune_model $zs_path $peft_path 1-1
bash ./scripts/evaluation_lorapruner_peft/eval_harness.sh $merged_prune_model $zs_path $peft_path 1
