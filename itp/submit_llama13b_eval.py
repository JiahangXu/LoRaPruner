import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path, pt, mrk = arg
    for prompt_type in [0, pt]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing.py submit --target sing_research --model_name eval_llama13b "
        command_list_no_prompt = [
            command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_piqa.sh --task_name piqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            # command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
        ]
        command_list = [
            command_head + f"--file evaluation_llama13b/eval_wikitext.sh --task_name wikitext2_eval --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_piqa.sh --task_name piqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_storycloze.sh --task_name storycloze --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_arcc.sh --task_name arc-c --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_arce.sh --task_name arc-e --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_winogrande.sh --task_name winogrande --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            # command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_obqa.sh --task_name obqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            # command_head + f"--file evaluation_llama13b/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
            # command_head + f"--file evaluation_llama13b/eval_gsm8k_train.sh --task_name gsm8k --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
            # command_head + f"--file evaluation_llama13b/eval_multiarith.sh --task_name multiarith --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4", 
        ]
        if prompt_type == 0:
            command_list = command_list_no_prompt + command_list
        for command in command_list:
            os.system(command)


waiting_jobs = [
    # llama-13b
    ("LoRaPruner/gpt4alpaca_llama13b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-3-21-48/epoch2/", "llama13b_8-3-21-48", 1), 
    
    
    ## ================= fix LR scheduler bug ================= ##
    # # mark 36
    # /mnt/data/LoRaPruner/gpt4alpaca_llama13b_prompt_nogate-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-23-5-24", "mark36_epoch5", 1),

    # # mark 37
    # ("LoRaPruner/gpt4alpaca_llama13b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-21-9-58", "mark37_epoch5", 1),
    
    # # mark 38
    # ("LoRaPruner/gpt4alpaca_llama13b_prompt_nogate-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-21-20-20", "mark38_epoch5", 2),

    # # mark 39
    # ("LoRaPruner/gpt4alpaca_llama13b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-21-20-21", "mark39_epoch5", 2),

    # # mark 40
    # ("LoRaPruner/gpt4alpaca_llama13b_prompt_nogate-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-21-19-9", "mark40_epoch5", 1),

    # # mark 41
    # ("LoRaPruner/gpt4alpaca_llama13b_closeinit_gate_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-21-19-7", "mark41_epoch5", 1),

    # # mark 42
    # ("LoRaPruner/gpt4alpaca_llama13b_prompt_nogate-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-21-23-8", "mark42_epoch5", 2),

    # # mark 43
    # ("LoRaPruner/gpt4alpaca_llama13b_closeinit_gate_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-21-23-8", "mark43_epoch5", 2),

    # # mark 44
    # ("LoRaPruner/gpt4alpaca_llama13b_prompt_nogate-s50.0-lr5e-05-reglr0.05-warmup4/2023-8-22-18-33", "mark44_epoch5", 1),

    # # mark 45
    # ("LoRaPruner/gpt4alpaca_llama13b_closeinit_gate_0.5lagST-s50.0-lr5e-05-reglr0.05-warmup4/2023-8-22-18-33", "mark45_epoch5", 1),

    # # mark 46
    # ("LoRaPruner/gpt4alpaca_llama13b_prompt_nogate-s50.0-lr5e-05-reglr0.05-warmup4/2023-8-21-23-8", "mark46_epoch5", 2),

    # # mark 47
    # ("LoRaPruner/gpt4alpaca_llama13b_closeinit_gate_0.5lagST-s50.0-lr5e-05-reglr0.05-warmup4/2023-8-21-23-8", "mark47_epoch5", 2),
]

args = []
for ckpt_path, mark, prompt_type in waiting_jobs:
    args.append((ckpt_path, prompt_type, mark))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


