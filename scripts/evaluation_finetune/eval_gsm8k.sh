
# # zero_shot_cot
# # python evaluation.py --output_dir ./ --model_name_or_path /home/jiahangxu/working/llama/7B_converted --eval_dataset_name multiarith --eval_method zero_shot_cot

# # few_shot_cot
# # python evaluation.py --output_dir ./ --model_name_or_path /home/jiahangxu/working/llama/7B_converted --eval_dataset_name addsub --eval_method few_shot_cot

baseline_pruned_model=/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-23/epoch4
pretrained_pruned_model=${1}
cp $baseline_pruned_model/zs.pt $pretrained_pruned_model/
cp $baseline_pruned_model/l0_module.pt $pretrained_pruned_model/

cp -r $baseline_pruned_model/llama_pruned ./

python evaluation.py \
    --output_dir ./ \
    --model_name_or_path ./llama_pruned \
    --pretrained_pruned_model $pretrained_pruned_model \
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
