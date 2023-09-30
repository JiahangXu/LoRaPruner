# 5.4B mark25  
python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name mark_25_5.4B_LoraApplyAll --file lora_apply_all_pruning/mark_25

# 5B 
# mark28-2 1-4-3
python run_sing.py submit --sparsity 0.2684 --target sing_octo --task_name gpt4alpaca --model_name mark_28-2_5B_LoraApplyAll --file lora_apply_all_pruning/mark_28-2
# mark28-4 1-4-2
python run_sing.py submit --sparsity 0.2684 --target sing_octo --task_name gpt4alpaca --model_name mark_28-4_5B_LoraApplyAll --file lora_apply_all_pruning/mark_28-4


# 4.5B
# mark28-5 1-4-2
python run_sing.py submit --sparsity 0.3456 --target sing_octo --task_name gpt4alpaca --model_name mark_28-5_4.5B_LoraApplyAll --file lora_apply_all_pruning/mark_28-5


# 4B 
# mark28-6 1-4-2
python run_sing.py submit --sparsity 0.423 --target sing_octo --task_name gpt4alpaca --model_name mark_28-6_4B_LoraApplyAll --file lora_apply_all_pruning/mark_28-6
