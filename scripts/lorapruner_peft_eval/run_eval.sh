# bash /home/jiahangxu/LoRaPruner/merge_and_eval.sh \
#     /home/jiahangxu/llama/pruned_llama_1ep_1e-5 \
#     ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6 \
#     0 0

CUDA_VISIBLE_DEVICES=0 python main.py bbh --model_name llama --model_path  ~/myfastnn/LoRaPruner/A100_peft_ckpt/mark25_lr5e-5 --prompt_mark 1
# nohup bash /home/jiahangxu/LoRaPruner/run_eval.sh >  eval_mark25_1.txt 2>&1 &


# bash /home/jiahangxu/LoRaPruner/merge_and_eval.sh \
#     /home/jiahangxu/llama/pruned_llama_1ep_1e-5 \
#     ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6 \
#     1-1


# bash /home/jiahangxu/LoRaPruner/merge_and_eval.sh \
#     /home/jiahangxu/llama/pruned_llama_1ep_2e-5 \
#     ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6 \
#     0

# bash /home/jiahangxu/LoRaPruner/merge_and_eval.sh \
#     /home/jiahangxu/llama/pruned_llama_1ep_2e-5 \
#     ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6 \
#     1-1



# bash /home/jiahangxu/LoRaPruner/merge_and_eval.sh \
#     /home/jiahangxu/llama/pruned_llama_new_prompt_new_zs_1ep \
#     ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7



# bash /home/jiahangxu/LoRaPruner/merge_and_eval.sh \
#     /home/jiahangxu/llama/pruned_llama_new_prompt_new_zs_2ep \
#     ~/myfastnn/LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7

# nohup bash run_eval.sh > eval_mark25_1.txt 2>&1 &