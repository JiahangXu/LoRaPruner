import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path, prompt_type, mark = arg
    command_head = "python run_sing.py submit --target sing_research --model_name eval_llama2-7b "
    command_list_no_prompt = [
        command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
        command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
    ]
    command_list = [
        # command_head + f"--file evaluation_llama2/eval_wikitext.sh --task_name wikitext2_eval --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
        command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_piqa.sh --task_name piqa --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
        command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_storycloze.sh --task_name storycloze --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num {2 if prompt_type != 0 else 1}",
        command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_arcc.sh --task_name arc-c --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
        command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_arce.sh --task_name arc-e --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num {4 if prompt_type != 0 else 2}",
        command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_winogrande.sh --task_name winogrande --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
        # command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
        command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_obqa.sh --task_name obqa --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
        # command_head + f"--file evaluation_llama2/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
        # command_head + f"--file evaluation_llama2/eval_gsm8k_train.sh --task_name gsm8k --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
        # command_head + f"--file evaluation_llama2/eval_multiarith.sh --task_name multiarith --ckpt_dir {ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1", 
    ]
    if prompt_type == 0:
        command_list = command_list_no_prompt + command_list
    for command in command_list:
        os.system(command)


waiting_jobs = [
      
    # llama2-7b
    ("/mnt/data/LoRaPruner/gpt4alpaca_llama2_7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-4-0-19/epoch4/", "llama2-7b_threshold0.5_lagST_epoch4_8-4-0-19_no_prompt", 0),
    ("/mnt/data/LoRaPruner/gpt4alpaca_llama2_7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-4-0-19/epoch4/", "llama2-7b_threshold0.5_lagST_epoch4_8-4-0-19_prompt_long", 1),

    # llama2-7b nogate
    ("/mnt/data/LoRaPruner/gpt4alpaca_llama2_7b_promptlong_nogate-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-7-4-23/epoch2/best/", "llama2-7b_epoch2_8-7-4-23_no_prompt", 0),
    ("/mnt/data/LoRaPruner/gpt4alpaca_llama2_7b_promptlong_nogate-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-7-4-23/epoch2/best/", "llama2-7b_epoch2_8-7-4-23_prompt_long", 1),
    
]

args = []
for ckpt_path, mark, prompt_type in waiting_jobs:
    args.append((ckpt_path, prompt_type, mark))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


