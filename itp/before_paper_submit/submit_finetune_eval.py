import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    base_model, ckpt_path, pt, mrk = arg
    for prompt_type in [0]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing_FT.py submit --target sing_octo --model_name eval_Finetuned_llama7b "
        command_list_no_prompt = [
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
        ]
        command_list = [
            # command_head + f"--file evaluation_finetune/eval_harness.sh --task_name harness --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_nqopen.sh --task_name nqopen --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_triviaqa.sh --task_name triviaqa --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_reasoning.sh --task_name reasoning --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_squad.sh --task_name squad --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_race.sh --task_name race --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_wikitext.sh --task_name wikitext2_eval --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_c4.sh --task_name c4 --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation_finetune/eval_mmlu.sh --task_name mmlu --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
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

    # # Finetune mark25FT9
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_5e-5-s30.0-lr5e-05-reglr0.05-warmup0/2023-9-3-4-2/epoch0",
    #  "mark25FT9", 1),

    # # Finetune mark25FT10
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark25_3e-5_qkvo-s30.0-lr3e-05-reglr0.05-warmup0/2023-9-3-9-38/epoch0",
    #  "mark25FT10", 1),
    
    # # Finetune mark25FT11
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark25_8e-6_qkvo-s30.0-lr8e-06-reglr0.05-warmup0/2023-9-3-13-55/epoch0",
    #  "mark25FT11", 1),
    
    # # Finetune mark25FT12
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-3-21-1/epoch0",
    #  "mark25FT12_epoch0", 1),
    
    # # Finetune mark25FT13
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_2e-4-s30.0-lr0.0002-reglr0.05-warmup0/2023-9-3-21-0/epoch0",
    #  "mark25FT13_epoch0", 1),
    
    # # Finetune mark25FT14
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_1e-4-s30.0-lr8e-05-reglr0.05-warmup0/2023-9-4-10-42/epoch1",
    #  "mark25FT14_epoch1", 1),
    
    # # Finetune mark25FT15
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_2e-4-s30.0-lr0.0002-reglr0.05-warmup0/2023-9-4-10-39/epoch0",
    #  "mark25FT15_epoch0", 1),

    # Finetune mark25FT16
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-4-10-41/epoch1",
    #  "mark25FT16_epoch1", 1),

    # Finetune mark25FT9-evallong
    ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
     "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_5e-5-s30.0-lr5e-05-reglr0.05-warmup0/2023-9-5-23-49/epoch0",
     "mark25FT9_evallong_epoch0", 1),
   
    # # Finetune mark25FT16-evallong
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-6-16-32/epoch1",
    #  "mark25FT16_evallong_epoch1", 1),
    
    # # Finetune mark25FT17-epoch0
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_5e-5-s30.0-lr5e-05-reglr0.05-warmup0/2023-9-7-15-15/epoch0",
    #  "mark25FT17_epoch0", 1),
    
    # # Finetune mark25FT17-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark25_5e-5-s30.0-lr5e-05-reglr0.05-warmup0/2023-9-7-15-15/epoch1",
    #  "mark25FT17_epoch1", 1),

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

    # # Finetune mark72FT1
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-14-15/epoch14",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark34-s30.0-lr8e-06-reglr0.05-warmup0/2023-8-30-23-45/epoch1",
    #  "mark34FT1", 1),

    #  Finetune mark72FT2
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-14-15/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-5-3-6/epoch0",
    #  "mark72FT2", 1),
    
    # # Finetune mark72FT2_best
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-14-15/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-5-3-6/best",
    #  "mark72FT2_best", 1),
    
    # # # Finetune mark72FT4_epoch0
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-14-15/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-5-22-36/epoch0",
    #  "mark72FT3_epoch0", 1),
    
    # # # Finetune mark72FT4_best
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-14-15/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-5-22-36/best",
    #  "mark72FT3_best", 1),
    
    # # # Finetune mark72FT4_epoch1
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-14-15/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr8e-05-reglr0.05-warmup0/2023-9-5-22-51/epoch1",
    #  "mark72FT4_epoch1", 1),
    
    # # # Finetune mark72FT4_best
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-14-15/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr8e-05-reglr0.05-warmup0/2023-9-5-22-51/best",
    #  "mark72FT4_best", 1),
     
     

    # # Finetune mark84FT1
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark84-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-0-48/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-5-5-24/epoch0",
    #  "mark34FT1", 1),

    # # Finetune mark84FT2
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark84-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-0-48/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-5-5-6/epoch0",
    #  "mark84FT2", 1),
    
    # # Finetune mark84FT2-best
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark84-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-0-48/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-5-5-6/best",
    #  "mark84FT2_best", 1),

    # # Finetune mark84FT3-best
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark84-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-0-48/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-5-22-52/best",
    #  "mark84FT3_best", 1),
    
    # # Finetune mark84FT3-epoch1
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark84-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-0-48/epoch14",
    #  "LoRaPruner/alpacagpt4_llama7b_promptlong_FTbased_mark87_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-5-22-52/epoch1",
    #  "mark84FT3_epoch1", 1),

    # # Finetune mark107FT1
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_mark107-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark107_2e-4-s30.0-lr0.0002-reglr0.05-warmup0/2023-9-12-10-30/epoch1",
    #  "mark107FT1", 1),

    # # Finetune mark108FT1
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_cubic_mark108-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark108_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-12-10-51/epoch1",
    #  "mark108FT1", 1),

    # # Finetune mark72LLMQATFT14_8e-5-epoch0
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-9-8-47/epoch14",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark72LLMQATFT16_8e-5-s30.0-lr8e-05-reglr0.05-warmup0/2023-9-10-12-32/epoch0",
    #  "mark72LLMQATFT14_epoch0", 1),

    # # Finetune mark72LLMQATFT14_8e-5-epoch1
    #  ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-9-8-47/epoch14",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark72LLMQATFT16_8e-5-s30.0-lr8e-05-reglr0.05-warmup0/2023-9-10-12-32/epoch1",
    #  "mark72LLMQATFT14_8e-5_epoch1", 1),
    
    # # Finetune mark72LLMQATFT14_8e-6-epoch0
    #  ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-9-8-47/epoch14",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark72LLMQATFT16_8e-6-s30.0-lr8e-06-reglr0.05-warmup0/2023-9-12-2-49",
    #  "mark72LLMQATFT14_8e-6_epoch0", 1),
     
    # # Finetune mark72LLMQATFT16_1e-4-epoch0
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-9-8-47/epoch14",
    #  "LoRaPruner/alpacaclean_llama7b_promptlong_FTbased_mark72LLMQATFT16_1e-4-s30.0-lr0.0001-reglr0.05-warmup0/2023-9-10-19-25/epoch1",
    # "mark72LLMQATFT16_1e-4_epoch1", 1),
     
    # # Finetune mark72LLMQATFT16_1e-5-epoch1
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-9-8-47/epoch14",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark72LLMQATFT16_1e-5-s30.0-lr1e-05-reglr0.05-warmup0/2023-9-11-8-23/epoch1/",
    #  "mark72LLMQATFT16_1e-5_epoch1", 1),

    # # Finetune mark84LLMQATFT16_4e-6-epoch1
    # ("LoRaPruner/llmpruner-5k_llama7b_promptlong_mark84-1-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-11-19-18/epoch14",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark84LLMQATPromptLong_4e-6-s30.0-lr4e-06-reglr0.05-warmup0/2023-9-16-6-20/epoch1",
    #  "mark84LLMQATFT1_4e-6_epoch1", 1),
    
    ######====================================######
    # mark 28-2
    # # Finetune mark28-2_4e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-13-4-15/epoch1/",
    #  "mark28-2_4e-4_epoch1", 1),
    
    # # Finetune mark28-2_8e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-13-2-47/epoch2/",
    #  "mark28-2_8e-4_epoch2", 1),
    
    # Finetune mark28-2_1.6e-3-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-13-3-48/epoch1/",
    #  "mark28-2_1.6e-3_epoch1", 1),

    # # Finetune mark28-2_1e-3-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_1e-3-s30.0-lr0.001-reglr0.05-warmup0/2023-9-16-8-5/epoch1/",
    #  "mark28-2_1e-3_epoch1", 1),

    # # Finetune mark28-2_2e-3-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_2e-3-s30.0-lr0.002-reglr0.05-warmup0/2023-9-16-8-6/epoch0/",
    #  "mark28-2_2e-3_epoch0", 1),
    # # Finetune mark28-2_2e-3-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_2e-3-s30.0-lr0.002-reglr0.05-warmup0/2023-9-16-8-6/epoch1/",
    #  "mark28-2_2e-3_epoch1", 1),

    # # Finetune mark28-2_3e-3-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_3e-3-s30.0-lr0.003-reglr0.05-warmup0/2023-9-16-8-18/epoch1/",
    #  "mark28-2_3e-3_epoch1", 1),

    # # Finetune mark28-2_1.6e-3Ep2-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-16-8-6/epoch1/",
    #  "mark28-2_1.6e-3Ep2_epoch1", 1),

    # # Finetune mark28-2_9e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
   #   "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_9e-4-s30.0-lr0.0009-reglr0.05-warmup0/2023-9-19-1-21/epoch1/",
   #   "mark28-2_9e-4_epoch1", 1),

     # # Finetune mark28-2_9e-8-epoch1
   #  ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7",
   #   "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_9e-8-s30.0-lr9e-08-reglr0.05-warmup0/2023-9-18-8-9/epoch2/",
   #   "mark28-2_9e-8_epoch2", 1),


   
    ######====================================######
    # mark 28-4
    # # Finetune mark28-4-epoch7_8e-4
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-14-0-30/epoch1/",
    #  "mark28-4_8e-4_epoch1Corrected", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-14-0-30/epoch2/",
    #  "mark28-4_8e-4_epoch2Corrected", 1),
    # # Finetune mark28-4-epoch7_8e-4
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-4_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-16-8-51/epoch0/",
    #  "mark28-4_8e-4_epoch0", 1),
    
    # # Finetune mark28-4-epoch7_1.6e-3
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-14-0-29/epoch1/",
    #  "mark28-4_1.6e-3_epoch1Corrected", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-14-0-29/epoch2/",
    #  "mark28-4_1.6e-3_epoch2Corrected", 1),
    
    # # # Finetune mark28-4-epoch7_4e-4
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-4_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-16-22-18/epoch1/",
    #  "mark28-4_4e-4_epoch1", 1),
    
    
    # # mark 28-4-ep6
    # # Finetune mark28-4-epoch6_8e-4
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_epoch5_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-15-9-4/epoch1/",
    #  "mark28-4_8e-4-ep6_epoch1", 1),
    # # Finetune mark28-4-epoch6_8e-4
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-4_epoch5_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-16-10-36/epoch1/",
    #  "mark28-4_8e-4-ep6_epoch1", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-14-1-38/epoch2/",
    #  "mark28-4_8e-4-ep6_epoch2", 1),
    
    # # Finetune mark28-4-epoch6_1.6e-3
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-14-2-3/epoch1/",
    #  "mark28-4-ep6_1.6e-3-ep6_epoch1", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-14-2-3/epoch2/",
    #  "mark28-4-ep6_1.6e-3-ep6_epoch2", 1),
    
    # # Finetune mark28-4-epoch6_4e-4
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_epoch5_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-15-9-34/epoch1/",
    #  "mark28-4_4e-4-ep6_epoch1", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-2_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-14-3-21/epoch2/",
    #  "mark28-4_4e-4-ep6_epoch2", 1),
   
    

    ######====================================######
    # mark 28-5
    # # Finetune mark28-5_8e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    # "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-13-11-34/epoch1/",
    # "mark28-5_8e-4_epoch1", 1),

    # # Finetune mark28-5_5e-4-epoch1
    #  ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_5e-4-s30.0-lr0.0005-reglr0.05-warmup0/2023-9-17-0-33/epoch1/",
    #  "mark28-5_5e-4_epoch1", 1),
    # # Finetune mark28-5_5e-4-epoch2
    #  ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_5e-4-s30.0-lr0.0005-reglr0.05-warmup0/2023-9-17-0-33/epoch2/",
    #  "mark28-5_5e-4_epoch2", 1),

    # # Finetune mark28-5_6e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_6e-4-s30.0-lr0.0006-reglr0.05-warmup0/2023-9-17-0-36/epoch1/",
    #  "mark28-5_6e-4_epoch1", 1),
    # # Finetune mark28-5_6e-4-epoch2
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_6e-4-s30.0-lr0.0006-reglr0.05-warmup0/2023-9-17-0-36/epoch2/",
    #  "mark28-5_6e-4_epoch2", 1),

    # Finetune mark28-5_4e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-13-17-56/epoch2/",
    #  "mark28-5_4e-4_epoch1", 1),

     # Finetune mark28-5_4e-4-epoch1
     #("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
      #"LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-13-17-56/epoch1/",
     # "mark28-5_4e-4_epoch2", 1),

    # Finetune mark28-5_4e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
      #"LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-13-17-56/epoch2/",
    #  "mark28-5_4e-4_epoch3", 1),
    
    # # Finetune mark28-5_3e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_3e-4-s30.0-lr0.0003-reglr0.05-warmup0/2023-9-17-8-51/epoch0/",
    #  "mark28-5_3e-4_epoch0", 1),

    # # # Finetune mark28-5_3e-4-epoch1
    #  ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #   "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_3e-4-s30.0-lr0.0003-reglr0.05-warmup0/2023-9-17-8-51/epoch1/",
    #   "mark28-5_3e-4_epoch1", 1),

    # # Finetune mark28-5_3e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_3e-4-s30.0-lr0.0003-reglr0.05-warmup0/2023-9-17-8-51/epoch2/",
    #  "mark28-5_3e-4_epoch2", 1),

    # # Finetune mark28-5_2e-4-epoch1
   #  ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_2e-4-s30.0-lr0.0002-reglr0.05-warmup0/2023-9-18-8-20/epoch2/",
    #  "mark28-5_2e-4_epoch2", 1),

    # # Finetune mark28-5_1.6e-3-epoch1
    #  ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #   "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-13-17-56/epoch2/",
    #   "mark28-5_1.6e-3_epoch2", 1),


    


    # Finetune mark28-5_4e-4-2eps-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-17-0-33/epoch1/",
    #  "mark28-5_4e-4Ep2_epoch1", 1),

    # # Finetune mark28-5_1.6e-3-epoch2
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-13-17-56/epoch2/",
    #  "mark28-5_1.6e-3_epoch2", 1),

    # # Finetune mark28-5-epoch5-8e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-14-9-38/epoch2/",
    #  "mark28-5_8e-ep5-4_epoch2", 1),

    #  # # Finetune mark28-5-epoch5_4e-4-epoch1
    #   ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch5",
    #    "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-14-17-23/epoch1/",
    #    "mark28-5_ep5-4e-4_epoch1", 1),

    # # Finetune mark28-5-epoch5_1.6e-3-epoch2
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch5",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-5_selected_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-14-11-56/epoch1/",
    #  "mark28-5_ep5-1.6e-3_epoch1", 1),


    ######====================================######
    # mark 28-6
    # Finetune mark28-6_8e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-7-7/epoch6",
    #   "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-6_selected_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-13-18-4/epoch1/",
    #   "mark28-6_8e-4_epoch1Corrected", 1),
    
    # Finetune mark28-6_1.6e-3-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-7-7/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-6_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-16-8-33/epoch0/",
    #  "mark28-6_1.6e-3_epoch0", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-7-7/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-6_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-16-8-33/epoch1/",
    #  "mark28-6_1.6e-3_epoch1", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-7-7/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-6_1.6e-3-s30.0-lr0.0016-reglr0.05-warmup0/2023-9-16-8-33/epoch2/",
    #  "mark28-6_1.6e-3_epoch2", 1),

    # Finetune mark28-6_4e-4-epoch1
    #("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-7-7/epoch6",
   #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-6_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-16-8-48/epoch0",
   #  "mark28-6_4e-4_epoch0", 1),
   # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-7-7/epoch6",
   #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-6_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-16-8-48/epoch1",
    # "mark28-6_4e-4_epoch1", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-7-7/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-6_4e-4-s30.0-lr0.0004-reglr0.05-warmup0/2023-9-16-8-48/epoch2",
    #  "mark28-6_4e-4_epoch2", 1),

    # Finetune mark28-6_8e-4-epoch1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-6-50/epoch6",
    #  "LoRaPruner/gpt4alpaca_llama7b_promptlong_FTbased_mark28-6_selected_8e-4-s30.0-lr0.0008-reglr0.05-warmup0/2023-9-13-18-4/epoch1/",
    #  "mark28-6_8e-4_epoch1", 1),

]

args = []
for base_model, ckpt_path, mark, prompt_type in waiting_jobs:
    args.append((base_model, ckpt_path, prompt_type, mark))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


