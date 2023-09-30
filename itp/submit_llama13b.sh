# 11B, layer gate sparsity 0.1594
# python run_sing.py submit --sparsity 0.1594 --target sing_octo --task_name gpt4alpaca --model_name mark_25_1-4-3_11B_llama13b --file llama13b_pruning/mark_25_1-4-3
python run_sing.py submit --sparsity 0.1594 --target sing_octo --task_name gpt4alpaca --model_name mark_25_1-4-2_11B_llama13b --file llama13b_pruning/mark_25_1-4-2
python run_sing.py submit --sparsity 0.1594 --target sing_octo --task_name gpt4alpaca --model_name mark_25_1-4-3_11B_llama13b_LoraApplyAll --file llama13b_lora_apply_all_pruning/mark_25_1-4-3
python run_sing.py submit --sparsity 0.1594 --target sing_octo --task_name gpt4alpaca --model_name mark_25_1-4-2_11B_llama13b_LoraApplyAll --file llama13b_lora_apply_all_pruning/mark_25_1-4-2

# # 10B, no gate sparsity 0.2382
# python run_sing.py submit --sparsity 0.2382 --target sing_octo --task_name gpt4alpaca --model_name mark_28-2_1-4-3_10B_llama13b --file llama13b_pruning/mark_28-2_1-4-3
# python run_sing.py submit --sparsity 0.2382 --target sing_octo --task_name gpt4alpaca --model_name mark_28-4_1-4-2_10B_llama13b --file llama13b_pruning/mark_28-4_1-4-2
# python run_sing.py submit --sparsity 0.2382 --target sing_octo --task_name gpt4alpaca --model_name mark_28-2_1-4-3_10B_llama13b_LoraApplyAll --file llama13b_lora_apply_all_pruning/mark_28-2_1-4-3
# python run_sing.py submit --sparsity 0.2382 --target sing_octo --task_name gpt4alpaca --model_name mark_28-4_1-4-2_10B_llama13b_LoraApplyAll --file llama13b_lora_apply_all_pruning/mark_28-4_1-4-2

# # 9B, no gate 0.317
# python run_sing.py submit --sparsity 0.317 --target sing_octo --task_name gpt4alpaca --model_name mark_28-5_1-4-3_9B_llama13b --file llama13b_pruning/mark_28-5_1-4-3
# python run_sing.py submit --sparsity 0.317 --target sing_octo --task_name gpt4alpaca --model_name mark_28-5_1-4-2_9B_llama13b --file llama13b_pruning/mark_28-5_1-4-2
# python run_sing.py submit --sparsity 0.317 --target sing_octo --task_name gpt4alpaca --model_name mark_28-5_1-4-3_9B_llama13b_LoraApplyAll --file llama13b_lora_apply_all_pruning/mark_28-5_1-4-3
# python run_sing.py submit --sparsity 0.317 --target sing_octo --task_name gpt4alpaca --model_name mark_28-5_1-4-2_9B_llama13b_LoraApplyAll --file llama13b_lora_apply_all_pruning/mark_28-5_1-4-2
