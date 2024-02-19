export PYTHONPATH='.'

base_model_path=$1
zs_path=$2
output_dir=$3

python merge_zs.py \
  --model_name_or_path $base_model_path \
  --pretrained_pruned_model $zs_path \
  --output_dir $output_dir

echo "MERGE ZS FINISHED"
