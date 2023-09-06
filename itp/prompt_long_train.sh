# git checkout prompt_long

# Alpaca cleaned
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name alpacaclean --model_name Aligned_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacacleaned_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv-closeinit-gate-thres0.5lagST

# # Llama7b, Cubic, warmup3, spar 30, long, layergate1
# python run_sing.py submit --sparsity 0.3 --target sing_research --task_name gpt4alpaca --model_name Aligned_CubicSpar_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup3-epoch6 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch6-warmup3-30%-qv-closeinit-gate-thres0.5lagST-CubicSpar
# sleep 60

# # Llama7b, Cubic, warmup3, spar 30, long, no gate
# python run_sing.py submit --sparsity 0.3 --target sing_research --task_name gpt4alpaca --model_name Aligned_CubicSpar_promptlong_nogate_llama7b-lr5e-05-reglr0.05-warmup3-epoch6 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch6-warmup3-30%-qv-CubicSpar
# sleep 60

# # Llama7b, Cubic, warmup4, spar 30, long, layergate1
# python run_sing.py submit --sparsity 0.3 --target sing_research --task_name gpt4alpaca --model_name Aligned_CubicSpar_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv-closeinit-gate-thres0.5lagST-CubicSpar
# sleep 60

# # Llama7b, Cubic, warmup4, spar 30, long, no gate
# python run_sing.py submit --sparsity 0.3 --target sing_research --task_name gpt4alpaca --model_name Aligned_CubicSpar_promptlong_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv-CubicSpar
# sleep 60

# # Llama7b, Linear, warmup4, spar 20, long, layergate1
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-20%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama7b, Linear, warmup4, spar 30, long, layergate1
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama7b, Linear, warmup4, spar 50, long, layergate1
# python run_sing.py submit --sparsity 0.5 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-50%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama7b, Linear, warmup4, spar 20, long, no gate
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-20%-qv
# sleep 60

# # Llama7b, Linear, warmup4, spar 30, long, no gate
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv
# sleep 60

# # Llama7b, Linear, warmup4, spar 50, long, no gate
# python run_sing.py submit --sparsity 0.5 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-50%-qv
# sleep 60


# # Llama13b, Linear, warmup4, spar 20, long, layergate1
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_layer_gate_initclose_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-20%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama13b, Linear, warmup4, spar 30, long, layergate1
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_layer_gate_initclose_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# Llama13b, Linear, warmup4, spar 50, long, layergate1
# python run_sing.py submit --sparsity 0.5 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_layer_gate_initclose_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-50%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama13b, Linear, warmup4, spar 20, long, no gate
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_nogate_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-20%-qv
# sleep 60

# # Llama13b, Linear, warmup4, spar 30, long, no gate
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_nogate_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv
# sleep 60

# Llama13b, Linear, warmup4, spar 50, long, no gate
# python run_sing.py submit --sparsity 0.5 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptlong_nogate_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-50%-qv


# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name alpacaclean --model_name Sampled5k_promptlong_nogate_llama7b-lr5e-05-reglr0.05-warmup10-epoch30 --file prompt_training_alpacacleaned/train_alpacacleaned5k_lr5e-5-reglr0.05-epoch30-warmup10-20%-qv
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name alpacaclean --model_name Sampled5k_promptlong_layer_gate_llama7b-lr5e-05-reglr0.05-warmup10-epoch30 --file prompt_training_alpacacleaned/train_alpacacleaned5k_lr5e-5-reglr0.05-epoch30-warmup10-30%-qv-closeinit-gate-thres0.5lagST
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name alpacaclean --model_name Sampled5k_promptlong_nogate_llama7b-lr5e-05-reglr0.05-warmup10-epoch30 --file prompt_training_alpacacleaned/train_alpacacleaned5k_lr5e-5-reglr0.05-epoch30-warmup10-30%-qv


python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name alpacaclean --model_name SelectLayer_nogate_llama13b-lr5e-05-reglr0.05-warmup4-epoch10 --file llama13b_training/train_alpacacleaned5k_lr5e-5-reglr0.05-epoch10-warmup4-20%-qv
python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name SelectLayer_nogate_llama13b-lr5e-05-reglr0.05-warmup4-epoch10 --file llama13b_training/train_alpacagpt4-5k_lr5e-5-reglr0.05-epoch10-warmup4-20%-qv
