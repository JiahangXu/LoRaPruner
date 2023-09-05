import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    base_model, ckpt_path, pt, mrk = arg
    for prompt_type in [0, "1-1", pt]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing_FT.py submit --target sing_research --model_name eval_Finetuned_llama7b "
        command_list_no_prompt = [
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
        ]
        command_list = [
            command_head + f"--file evaluation_finetune/eval_harness.sh --task_name harness --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation_finetune/eval_wikitext.sh --task_name wikitext2_eval --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation_finetune/eval_c4.sh --task_name c4 --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
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
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-24-0-33/epoch1/", "markFT12_epoch1", 1),
    
    # # Finetune mark24
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-18-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-18-46/epoch1",
    #  "mark24FT", 1),

    # # Finetune mark25FT1
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-30-14-7/epoch1",
    #  "mark25FT", 1),
    
    # # Finetune mark25FT2
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-30-20-22/epoch1",
    #  "mark25FT2", 1),
    
    # # Finetune mark25FT3
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-30-23-6/epoch1",
    #  "mark25FT3", 1),
    
    # # Finetune mark25FT4
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25-s30.0-lr5e-06-reglr0.05-warmup0/2023-8-31-19-47/epoch0",
    #  "mark25FT4", 1),
    
    # # Finetune mark25FT5
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25-s30.0-lr2e-06-reglr0.05-warmup0/2023-8-31-20-18/epoch0",
    #  "mark25FT5", 1),
    
    # # Finetune mark25FT6
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25-s30.0-lr1e-06-reglr0.05-warmup0/2023-8-31-23-27/epoch0",
    #  "mark25FT6", 1),
    
    # Finetune mark25FT7
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_mark25-s30.0-lr1e-06-reglr0.05-warmup0/2023-8-31-22-47/epoch0",
    #  "mark25FT7", 1),

    # Finetune mark25FT8
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_8e-5-s30.0-lr8e-05-reglr0.05-warmup0/2023-9-3-4-2/epoch0",
    #  "mark25FT8", 1),

    # Finetune mark25FT9
    ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
     "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_5e-5-s30.0-lr5e-05-reglr0.05-warmup0/2023-9-3-4-2/epoch0",
     "mark25FT9", 1),

    # # Finetune mark25FT10
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark25_3e-5_qkvo-s30.0-lr3e-05-reglr0.05-warmup0/2023-9-3-9-38/epoch0",
    #  "mark25FT10", 1),
    
    # # Finetune mark25FT11
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark25_8e-6_qkvo-s30.0-lr8e-06-reglr0.05-warmup0/2023-9-3-13-55/epoch0",
    #  "mark25FT11", 1),

    # # Finetune mark56-1FT1
    # ("LoRaPruner/gptcleaned5k_llama7b_epoch30_warmup10_spar0.2_middle-s20.0-lr5e-05-reglr0.05-warmup10/2023-8-30-0-16/epoch29",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark56-1_5e-5-s30.0-lr5e-05-reglr0.05-warmup0/2023-9-3-5-21/epoch0",
    #  "mark56-1FT1", 1),

    # # Finetune mark56-1FT2
    # ("LoRaPruner/gptcleaned5k_llama7b_epoch30_warmup10_spar0.2_middle-s20.0-lr5e-05-reglr0.05-warmup10/2023-8-30-0-16/epoch29",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark56-1_3e-5-s30.0-lr3e-05-reglr0.05-warmup0/2023-9-3-5-18/epoch0",
    #  "mark56-1FT2", 1),


    # # Finetune mark56-1FT3
    # ("LoRaPruner/gptcleaned5k_llama7b_epoch30_warmup10_spar0.2_middle-s20.0-lr5e-05-reglr0.05-warmup10/2023-8-30-0-16/epoch29",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark56-1_8e-5-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-3-5-18/epoch0",
    #  "mark56-1FT3", 1),


    # # Finetune mark26FT1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-20-28/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark26_epoch4-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-30-23-12/epoch1",
    #  "mark26FT1", 1),
    
    # # Finetune mark27FT1
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-20-17/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark27-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-30-23-45/epoch1",
    #  "mark27FT1", 1),
    
    # # Finetune mark28FT1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-20-18-47/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark28-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-30-23-46/epoch1",
    #  "mark28FT1", 1),
    
    # # Finetune mark48FT1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-21-21-2/epoch5",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark48-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-30-23-46/epoch1",
    #  "mark48FT1", 1),
    
    # # Finetune mark34FT1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s50.0-lr5e-05-reglr0.05-warmup4/2023-8-20-20-31/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark34-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-30-23-45/epoch1",
    #  "mark34FT1", 1),

]

args = []
for base_model, ckpt_path, mark, prompt_type in waiting_jobs:
    args.append((base_model, ckpt_path, prompt_type, mark))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


