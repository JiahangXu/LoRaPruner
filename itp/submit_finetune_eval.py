import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path, pt, mrk = arg
    for prompt_type in [0, pt]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing.py submit --target sing_research --model_name eval_Finetuned_llama7b "
        command_list_no_prompt = [
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
        ]
        command_list = [
            command_head + f"--file evaluation_finetune/eval_harness.sh --task_name harness --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation_finetune/eval_wikitext.sh --task_name wikitext2_eval --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_piqa.sh --task_name piqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_storycloze.sh --task_name storycloze --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num {2 if prompt_type != 0 else 1}",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_arcc.sh --task_name arc-c --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_arce.sh --task_name arc-e --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num {4 if prompt_type != 0 else 2}",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_winogrande.sh --task_name winogrande --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_obqa.sh --task_name obqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
            # command_head + f"--file evaluation_finetune/eval_gsm8k_train.sh --task_name gsm8k --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
            # command_head + f"--file evaluation_finetune/eval_multiarith.sh --task_name multiarith --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1", 
        ]
        if prompt_type == 0:
            command_list = command_list_no_prompt + command_list
        for command in command_list:
            os.system(command)


waiting_jobs = [

    # origin model path
    # /mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-23/epoch4

    # Finetune mark11 7-31-21-23
    # mark FT1
    # ("/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr1e-06-reglr0.05-warmup0/2023-8-15-6-20/epoch0/", "8-15-6-20_epoch0_no_prompt", 0),
    # ("/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr1e-06-reglr0.05-warmup0/2023-8-15-6-20/epoch0/", "8-15-6-20_epoch0_prompt_long", 1),

    # mark FT2
    # ("/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr1e-06-reglr0.05-warmup0/2023-8-15-13-14/epoch2/", "8-15-13-14-FT3_epoch2_no_prompt", 0),
    # ("/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr1e-06-reglr0.05-warmup0/2023-8-15-13-14/epoch0/", "8-15-13-14-FT3_epoch2_prompt_long", 1),
    
    # mark FT3
    # ("/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-06-reglr0.05-warmup0/2023-8-15-13-14/epoch0/", "8-15-13-14-FT1_epoch2_no_prompt", 0),
    # ("/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-06-reglr0.05-warmup0/2023-8-15-13-14/epoch0/", "8-15-13-14-FT1_epoch2_prompt_long", 1),
    
    # mark FT4
    # ("/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-06-reglr0.05-warmup0/2023-8-15-13-13/epoch2/", "8-15-13-13_epoch2_no_prompt", 0),
    # ("/mnt/data/LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-06-reglr0.05-warmup0/2023-8-15-13-13/epoch2/", "8-15-13-13_epoch2_prompt_long", 1),
    
    # Finetune mark10 7-31-21-32
    # mark FT12
    ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-24-0-33/epoch1/", "markFT12_epoch1", 1),
    
]

args = []
for ckpt_path, mark, prompt_type in waiting_jobs:
    args.append((ckpt_path, prompt_type, mark))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


