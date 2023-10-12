import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path, pt, mrk, lora_param = arg
    for prompt_type in [0]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing.py submit --target sing_octo --model_name eval_llama7b "
        command_list = [
            # command_head + f"--file evaluation/eval_harness.sh --task_name harness --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --lora_param {lora_param} --node_num 1",
            # command_head + f"--file evaluation/eval_wikitext.sh --task_name wikitext2_eval --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --lora_param {lora_param} --node_num 1",
            command_head + f"--file evaluation/eval_mmlu.sh --task_name mmlu --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --lora_param {lora_param} --node_num 1",
            # command_head + f"--file evaluation/eval_bbh.sh --task_name bbh --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --lora_param {lora_param} --node_num 1",
        ]
        for command in command_list:
            os.system(command)


waiting_jobs = [
    ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6", "mark25_epoch6", 1, "Q.V"),
    # ("LoRaPruner/ablation_Study_loraapplyall_v1-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-18-9-32/epoch6", "Ablation_mark25_EPed_LoRaApplyAll_epoch6", 1, "Q.K.V.O.F"),
    # ("LoRaPruner/mark25_LoRaApplyAll-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-29-7-31/epoch6", "mark25_LoRaApplyAll_epoch6", 1, "Q.K.V.O.F"),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-21-22-53/epoch7", "mark28-2_EPed_epoch7", 1, "Q.V"),
    # ("LoRaPruner/mark28-2_LoRaApplyAll-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-30-0-17/epoch7", "mark28-2_LoRaApplyAll_epoch7", 1, "Q.K.V.O.F"),
    
    # ("LoRaPruner/mark28-4_LoRaApplyAll-s26.83-lr5e-05-reglr0.05-warmup4/2023-10-4-6-37/epoch6", "mark28-4_LoRaApplyAll_epoch6", 1, "Q.K.V.O.F"),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-21-23-12/epoch6", "mark28-5_1-4-2_EPed_epoch6", 1, "Q.V"),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-21-23-12/epoch7", "mark28-5_1-4-3_EPed_epoch7", 1, "Q.V"),
    # ("LoRaPruner/mark28-5_LoRaApplyAll-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-30-0-18/epoch6", "mark28-5_LoRaApplyAll_epoch6", 1, "Q.K.V.O.F"),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-21-23-14/epoch6", "mark28-6_1-4-2_EPed_epoch6", 1, "Q.V"),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_1-4-3-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-21-23-14/epoch7", "mark28-6_1-4-3_EPed_epoch7", 1, "Q.V"),
    # ("LoRaPruner/mark28-6_LoRaApplyAll-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-30-0-18/epoch6", "mark28-6_LoRaApplyAll_epoch6", 1, "Q.K.V.O.F"),
]    

args = []
for ckpt_path, mark, prompt_type, lora_param in waiting_jobs:
    args.append((ckpt_path, prompt_type, mark, lora_param))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


