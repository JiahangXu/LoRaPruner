
# # zero_shot_cot
# # python evaluation.py --output_dir ./ --model_name_or_path /home/jiahangxu/working/llama/7B_converted --eval_dataset_name multiarith --eval_method zero_shot_cot

# # few_shot_cot
# # python evaluation.py --output_dir ./ --model_name_or_path /home/jiahangxu/working/llama/7B_converted --eval_dataset_name addsub --eval_method few_shot_cot

python evaluation.py \
    --output_dir ./ \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --pretrained_pruned_model ${1} \
    --eval_dataset_name gsm8k \
    --validation_file /mnt/data/LPM/math_eval/gsm8k_test.json \
    --eval_method few_shot_cot \
    --use_lora True \
    --lora_rank 8 \
    --lora_train_bias none \
    --lora_alpha 8.0 \
    --lora_param Q.V \
    --lora_layers 32 \
    --eval_prompt_type ${2} \
