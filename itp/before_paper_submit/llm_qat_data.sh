# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name gpt4alpaca --model_name mark_72_LLMQAT_AddEval_SelectLayer_nogate_llama7b-lr5e-05-reglr0.05-warmup10-epoch15 --file select_layer_pruning_long/mark_72_llmqat

# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name llm_qat --model_name mark_84_LLMQAT_AddEval_SelectLayer_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup10-epoch15 --file select_layer_pruning_long/mark_84_llmqat
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name llm_qat --model_name mark_84_LLMQAT_GradAccm8_SelectLayer_Cubic_nogate_llama7b-lr4e-4-reglr0.05-warmup10-epoch15 --file select_layer_pruning_long/mark_84_llmqat_GradAccm8
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name llm_qat --model_name mark_84_LLMQAT_PromptLong_SelectLayer_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup10-epoch15 --file select_layer_pruning_long/mark_84_llmqat


# python run_sing.py submit --sparsity 0.3456 --target sing_octo --task_name llm_qat --model_name mark_125_LLMQAT_AddEval_SelectLayer_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup20-epoch25 --file select_layer_pruning_346/mark_125_llmqat

# python run_sing.py submit --sparsity 0.2684 --target sing_octo --task_name llm_qat --model_name mark_113_LLMQAT_AddEval_SelectLayer_Cubic_nogate_llama7b-lr5e-05-reglr0.05-warmup14-epoch20 --file select_layer_pruning_268/mark_113_llmqat


# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name llm_qat --model_name mark_25_LLMQAT_Aligned_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup4-epoch7 --file prompt_training/mark_25_llmqat
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name llm_qat --model_name mark_25_LLMQAT_Aligned_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup5-epoch7 --file prompt_training/mark_25_llmqat
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name llm_qat --model_name mark_25_LLMQAT_SelectLayer_Aligned_promptlong-lr5e-05-reglr0.05-warmup5-epoch7 --file prompt_training/mark_25_llmqat_selected

# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name llm_qat --model_name mark_25_LLMQAT96k_Aligned_promptlong_layer_gate_initclose_llama7b-lr5e-05-reglr0.05-warmup5-epoch7 --file prompt_training/mark_25_llmqat96k
# python run_sing.py submit --sparsity 0.2 --target sing_octo --task_name llm_qat --model_name mark_25_LLMQAT96k_SelectLayer_Aligned_promptlong-lr5e-05-reglr0.05-warmup5-epoch7 --file prompt_training/mark_25_llmqat96k_selected
