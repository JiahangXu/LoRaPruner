export CUDA_VISIBLE_DEVICES=7

# 5.4B
# base_model_path=$TEAMDRIVE/LoRaPruner/llama7b-2_mark25_LoRaApplyAll-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-29-7-14/epoch6
# bash ./scripts/merge_weights/merge_weights.sh $base_model_path ./llama_pruned_5.4B
base_model_path=$TEAMDRIVE/LoRaPruner/A100_pruning_ckpt/llama7b-2_mark25_LoRaApplyAll-s20.0-lr5e-05-reglr0.05-warmup4/2024-2-4-2-51/epoch6
bash ./scripts/merge_weights/merge_weights.sh $base_model_path ./llama_pruned_a100_5.4B

nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_5.4B 2e-4 a100_updatetokenizer_mark25_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_updatetokenizer_5.4B_2e-4.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_5.4B 1e-4 a100_updatetokenizer_mark25_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_updatetokenizer_5.4B_1e-4.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_5.4B 1e-5 a100_updatetokenizer_mark25_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_updatetokenizer_5.4B_1e-5.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_5.4B 5e-6 a100_updatetokenizer_mark25_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_updatetokenizer_5.4B_5e-6.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_5.4B 2e-5 a100_updatetokenizer_mark25_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_updatetokenizer_5.4B_2e-5.txt 2>&1 &


# 5B
# base_model_path=$TEAMDRIVE/LoRaPruner/llama7b-2_mark28-2_LoRaApplyAll-s26.83-lr5e-05-reglr0.05-warmup4/2023-10-5-7-19/epoch7
# bash ./scripts/merge_weights/merge_weights.sh $base_model_path ./llama_pruned_5B
base_model_path=$TEAMDRIVE/LoRaPruner/A100_pruning_ckpt/llama7b-2_mark28-2_LoRaApplyAll-s26.83-lr5e-05-reglr0.05-warmup4/2024-2-3-19-33/epoch7
bash ./scripts/merge_weights/merge_weights.sh $base_model_path ./llama_pruned_a100_5B

nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_5B 2e-4 a100_updatetokenizer_mark28-2_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_a100_5B_2e-4.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_5B 1e-4 a100_updatetokenizer_mark28-2_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_a100_5B_1e-4.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_5B 5e-5 a100_updatetokenizer_mark28-2_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_a100_5B_5e-5.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_5B 1e-5 a100_updatetokenizer_mark28-2_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_a100_5B_1e-5.txt 2>&1 &

# bash ./scripts/evaluation_lorapruner_peft/eval_all.sh ./llama_pruned_5B $base_model_path ./finetune_results/mark28-2_LoRaApplyAll_lr1e-5_bs8_AlpacaGPT4

# 4.5B
# base_model_path=$TEAMDRIVE/LoRaPruner/llama7b-2_mark28-5_LoRaApplyAll-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-29-22-16/epoch6
# bash ./scripts/merge_weights/merge_weights.sh $base_model_path ./llama_pruned_4.5B
base_model_path=$TEAMDRIVE/LoRaPruner/A100_pruning_ckpt/llama2-7b_mark28-5-s34.56-lr5e-05-reglr0.05-warmup4/2024-2-6-15-58/epoch6
bash ./scripts/merge_weights/merge_weights.sh $base_model_path ./llama_pruned_a100_4.5B

nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_4.5B 2e-4 a100_updatetokenizer_mark28-5 $base_model_path > ./finetune_log/finetune_log_a100_4.5B_2e-4.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_4.5B 1e-4 a100_updatetokenizer_mark28-5 $base_model_path > ./finetune_log/finetune_log_a100_4.5B_1e-4.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_4.5B 5e-5 a100_updatetokenizer_mark28-5 $base_model_path > ./finetune_log/finetune_log_a100_4.5B_5e-5.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_4.5B 1e-5 a100_updatetokenizer_mark28-5 $base_model_path > ./finetune_log/finetune_log_a100_4.5B_1e-5.txt 2>&1 &


# base_model_path=$TEAMDRIVE/LoRaPruner/A100_pruning_ckpt/llama7b-2_mark28-5_LoRaApplyAll-s34.56-lr5e-05-reglr0.05-warmup4/2024-2-11-0-25/epoch6
# bash ./scripts/merge_weights/merge_weights.sh $base_model_path ./llama_pruned_a100_4.5B_LoRaApplyAll

nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_4.5B_LoRaApplyAll 2e-4 a100_updatetokenizer_mark28-5_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_a100_4.5B_LoRaApplyAll_2e-4.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_4.5B_LoRaApplyAll 1e-4 a100_updatetokenizer_mark28-5_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_a100_4.5B_LoRaApplyAll_1e-4.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_4.5B_LoRaApplyAll 5e-5 a100_updatetokenizer_mark28-5_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_a100_4.5B_LoRaApplyAll_5e-5.txt 2>&1 &
nohup bash ./scripts/finetune/FT_llmpruner.sh ./llama_pruned_a100_4.5B_LoRaApplyAll 1e-5 a100_updatetokenizer_mark28-5_LoRaApplyAll $base_model_path > ./finetune_log/finetune_log_a100_4.5B_LoRaApplyAll_1e-5.txt 2>&1 &





# nohup bash run_FT.sh  > finetune_log_5B.txt 2>&1 & 
# 3953612