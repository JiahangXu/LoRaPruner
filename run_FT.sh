# bash FT_llmpruner.sh 1e-5 mark25 ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6
# bash FT_llmpruner.sh 5e-5 mark25 ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6
# bash FT_llmpruner.sh 1e-4 mark25 ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6

# base_model_path=~/working2/myfastnn/LoRaPruner/llama7b-2_mark25_LoRaApplyAll-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-29-7-14/epoch6
# bash FT_llmpruner.sh 1e-4 mark25_LoRaApplyAll $base_model_path
# bash FT_llmpruner.sh 1e-5 mark25_LoRaApplyAll $base_model_path
# bash FT_llmpruner.sh 5e-6 mark25_LoRaApplyAll $base_model_path
# bash FT_llmpruner.sh 2e-5 mark25_LoRaApplyAll $base_model_path
# bash FT_llmpruner.sh 8e-6 mark25_LoRaApplyAll $base_model_path

# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr1e-4_bs8_AlpacaGPT4 1
# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr1e-5_bs8_AlpacaGPT4 1
# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr5e-6_bs8_AlpacaGPT4 1
# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr2e-5_bs8_AlpacaGPT4 1
# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr8e-6_bs8_AlpacaGPT4/checkpoint-8000 1

# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr1e-4_bs8_AlpacaGPT4 0
# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr1e-5_bs8_AlpacaGPT4 0
# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr5e-6_bs8_AlpacaGPT4 0
# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr2e-5_bs8_AlpacaGPT4 0
# bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark25_LoRaApplyAll_lr8e-6_bs8_AlpacaGPT4/checkpoint-8000 0

base_model_path=~/working2/myfastnn/LoRaPruner/llama7b-2_mark28-2_LoRaApplyAll-s26.83-lr5e-05-reglr0.05-warmup4/2023-10-5-7-19/epoch7
# bash merge_weights.sh $base_model_path

bash FT_llmpruner.sh 2e-4 mark28-2_LoRaApplyAll $base_model_path
bash FT_llmpruner.sh 1e-4 mark28-2_LoRaApplyAll $base_model_path
bash FT_llmpruner.sh 1e-5 mark28-2_LoRaApplyAll $base_model_path

bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark28-2_LoRaApplyAll_lr2e-4_bs8_AlpacaGPT4 1
bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark28-2_LoRaApplyAll_lr1e-4_bs8_AlpacaGPT4 1
bash scripts/evaluation_llama2/eval_harness.sh $base_model_path ./finetune_results/mark28-2_LoRaApplyAll_lr1e-5_bs8_AlpacaGPT4 1



# nohup bash run_FT.sh  > finetune_log_5B.txt 2>&1 & 
# 3953612