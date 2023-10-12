# 20%： mark25，  
python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name mark_25_EPed_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-20%-qv-closeinit-gate-thres0.5lagST

# 26%, mark 28-2: 
# 1-4-3, 1-4-2;  
python run_sing.py submit --sparsity 0.2684 --target sing_octo --task_name gpt4alpaca --model_name mark_28-2_EPed_1-4-3_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch8 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch8-warmup4-26.8%-qv-CubicSpar
python run_sing.py submit --sparsity 0.2684 --target sing_octo --task_name gpt4alpaca --model_name mark_28-4_EPed_1-4-2_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file select_layer_pruning_268/mark_28-4

# 34%:mark 28-5; 
# 1-4-2 and 1-4-3; 
python run_sing.py submit --sparsity 0.3456 --target sing_octo --task_name gpt4alpaca --model_name mark_28-5_EPed_1-4-2_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file select_layer_pruning_346/mark_28-5
python run_sing.py submit --sparsity 0.3456 --target sing_octo --task_name gpt4alpaca --model_name mark_28-5_EPed_1-4-3_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch8 --file select_layer_pruning_346/mark_28-5_1-4-3


# 40% mark 28-6: 
# 1-4-2 and 1-4-3
python run_sing.py submit --sparsity 0.423 --target sing_octo --task_name gpt4alpaca --model_name mark_28-6_EPed_1-4-2_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file select_layer_pruning_423/mark_28-6
python run_sing.py submit --sparsity 0.423 --target sing_octo --task_name gpt4alpaca --model_name mark_28-6_EPed_1-4-3_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch8 --file select_layer_pruning_423/mark_28-6_1-4-3
