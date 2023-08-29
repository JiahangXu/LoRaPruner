
# git checkout prompt_middle

# # Llama7b, Linear, warmup4, spar 20, long, layergate1
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-20%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama7b, Linear, warmup4, spar 30, long, layergate1
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama7b, Linear, warmup4, spar 50, long, layergate1
# python run_sing.py submit --sparsity 0.5 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-50%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama7b, Linear, warmup4, spar 20, long, no gate
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-20%-qv
# sleep 60

# # Llama7b, Linear, warmup4, spar 30, long, no gate
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv
# sleep 60

# # Llama7b, Linear, warmup4, spar 50, long, no gate
# python run_sing.py submit --sparsity 0.5 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-50%-qv
# sleep 60


# # Llama13b, Linear, warmup4, spar 20, long, layergate1
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_layer_gate_initclose_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-20%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama13b, Linear, warmup4, spar 30, long, layergate1
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_layer_gate_initclose_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama13b, Linear, warmup4, spar 50, long, layergate1
# python run_sing.py submit --sparsity 0.5 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_layer_gate_initclose_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-50%-qv-closeinit-gate-thres0.5lagST
# sleep 60

# # Llama13b, Linear, warmup4, spar 20, long, no gate
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_nogate_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-20%-qv
# # sleep 60

# # Llama13b, Linear, warmup4, spar 30, long, no gate
# python run_sing.py submit --sparsity 0.3 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_nogate_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-30%-qv
# sleep 60

# # Llama13b, Linear, warmup4, spar 50, long, no gate
# python run_sing.py submit --sparsity 0.5 --target sing_octo --task_name gpt4alpaca --model_name Aligned_promptmiddle_nogate_llama13b-lr5e-05-reglr0.05-warmup4-epoch7 --file llama13b_training/train_alpacagpt4_lr5e-5-reglr0.05-epoch7-warmup4-50%-qv


# # Llama7b, linear, spar 20, middle, no gate, 4-30 layer
# python run_sing.py submit --sparsity 0.2 --target sing_research --task_name alpacaclean --model_name Test_nogate_llama7b-lr5e-05-reglr0.05-warmup4-epoch8 --file prompt_training_alpacacleaned/train_alpacacleaned_lr5e-5-reglr0.05-epoch8-warmup4-20%-qv

python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name alpacaclean --model_name Sampled5k_promptmiddle_nogate_llama7b-lr5e-05-reglr0.05-warmup10-epoch30 --file prompt_training_alpacacleaned/train_alpacacleaned5k_lr5e-5-reglr0.05-epoch30-warmup10-20%-qv
