base_model_path=$1
peft_path=$2
output_dir=$3

python merge_peft.py \
    --model_name_or_path $base_model_path \
    --peft_path $peft_path \
    --output_dir $output_dir
