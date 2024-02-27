source /home/aisilicon/miniconda3/bin/activate compresso
which python

pruned_path=$1
result_mark=$2

bash ./scripts/evaluation_lorapruner/eval_harness.sh $pruned_path $result_mark 0
bash ./scripts/evaluation_lorapruner/eval_harness.sh $pruned_path $result_mark 1-1
bash ./scripts/evaluation_lorapruner/eval_harness.sh $pruned_path $result_mark 1
