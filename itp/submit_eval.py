import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path, pt, mrk = arg
    for prompt_type in [0, "1-1", pt]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing.py submit --target sing_octo --model_name eval_llama7b "
        command_list_no_prompt = [
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
        ]
        command_list = [
            command_head + f"--file evaluation/eval_harness.sh --task_name harness --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_nqopen.sh --task_name nqopen --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_triviaqa.sh --task_name triviaqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_reasoning.sh --task_name reasoning --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_wikitext.sh --task_name wikitext2_eval --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_c4.sh --task_name c4 --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_mmlu.sh --task_name mmlu --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_bbh.sh --task_name bbh --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation/eval_race_high.sh --task_name race_high --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_race_middle.sh --task_name race_middle --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_piqa.sh --task_name piqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_storycloze.sh --task_name storycloze --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num {2 if prompt_type != 0 else 1}",
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_arcc.sh --task_name arc-c --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_arce.sh --task_name arc-e --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num {4 if prompt_type != 0 else 2}",
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_winogrande.sh --task_name winogrande --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # # command_head + f"--file evaluation/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_obqa.sh --task_name obqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
            # command_head + f"--file evaluation/eval_gsm8k_train.sh --task_name gsm8k --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
            # command_head + f"--file evaluation/eval_multiarith.sh --task_name multiarith --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1", 
        ]
        if prompt_type == 0:
            command_list = command_list_no_prompt + command_list
        for command in command_list:
            os.system(command)


waiting_jobs = [
    # # w/o layer gate
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-20-4-15/epoch4/", "7-20-4-15_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-20-4-15/epoch4/", "7-20-4-15_eval_prompt_long", 1),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-19-23-53/epoch4/", "prompt_7-19-23-53_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-19-23-53/epoch4/", "prompt_7-19-23-53_eval_prompt", 1),
    
    # ("LoRaPruner/promptmiddle_in_finetune_alpacagpt4_llama7b-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-21-0-32/epoch4/", "prompt_epoch4_7-21-0-32_no_prompt", 0),
    # ("LoRaPruner/promptmiddle_in_finetune_alpacagpt4_llama7b-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-21-0-32/epoch4/", "prompt_7-21-0-32_eval_prompt_middle", 2),
    
    # ("LoRaPruner/promptmiddle_in_warmup_alpacagpt4_llama7b-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-21-1-32/epoch4/", "prompt_warmup_epoch4_7-21-1-32_no_prompt", 0),
    # ("LoRaPruner/promptmiddle_in_warmup_alpacagpt4_llama7b-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-21-1-32/epoch4/", "prompt_warmup_7-21-1-32_eval_prompt_middle", 2),
    
    # compare mark-10 result
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-32/epoch4/", "prompt_epoch4_mark10-v1_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-32/epoch4/", "prompt_epoch4_7-31-21-32_eval_prompt_long", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-10-2-28/epoch4", "threshold0.5_lagST_epoch4_8-10-2-28_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-10-2-28/epoch3/best", "threshold0.5_lagST_epoch3_best_8-10-2-28_no_prompt", 0),
    # # mark 10 - reproduce
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-16-10-19/epoch4/", "prompt_epoch4_mark10-v2_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-16-22-49/epoch4/", "prompt_epoch4_mark10-v3_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-10-2-28/epoch4/", "prompt_epoch4_mark10-v4_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-16-10-19/epoch4/", "prompt_epoch4_8-16-10-19_prompt_long", 1),
    # # mark 10-1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-17-0-42/epoch4/", "prompt_epoch4_mark10-1_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-17-0-42/epoch4/", "prompt_epoch4_mark10-1_prompt_long", 1),
    # # mark 10-2
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-17-4-16/epoch4/", "prompt_epoch4_mark10-2_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-17-4-16/epoch4/", "prompt_epoch4_mark10-2_prompt_long", 1),
    # # mark 10-3
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-17-9-49/epoch4/", "prompt_epoch4_mark10-3_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST_resubmit-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-17-9-49/epoch4/", "prompt_epoch4_mark10-3_prompt_long", 1),

    # before fix lr scheduler bug
    # ("LoRaPruner/gpt4alpaca_llama7b_wolayer_layergate-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-11-9-59/epoch4/", " layer_gate_7-11-9-59_eval_prompt", 1),
    # ("LoRaPruner/gpt4alpaca_llama7b_wolayer_layergate-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-12-3-1/epoch4/", " layer_gate_7-12-3-1_eval_prompt", 1),

    # ("LoRaPruner/c4_llama7b_wolayer_layergate-s30.0-lr5e-05-reglr0.1-warmup10/2023-7-11-9-56/epoch19/", " layer_gate_7-11-9-56_eval_prompt", 1),
    # ("LoRaPruner/c4_llama7b_wolayer_layergate-s30.0-lr5e-05-reglr0.1-warmup10/2023-7-18-0-6/epoch19/", " layer_gate_7-18-0-6_eval_prompt", 1),

    # after debug layer gate (lagST, threshold 0.5)
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-28-1-50/epoch1/", "threshold0.5_lagST_7-28-1-50_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-28-1-50/epoch1/", "threshold0.5_lagST_7-28-1-50_prompt_long", 1),

    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-30-1-15/epoch4/best", "threshold0.5_lagST_epoch4_7-30-1-15_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-30-1-15/epoch4/", "threshold0.5_lagST_epoch4_7-30-1-15_prompt_long", 1),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_promptmiddle_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-30-1-12/epoch4/", "threshold0.5_lagST_epoch4_7-30-1-12_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptmiddle_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-30-1-12/epoch4/", "threshold0.5_lagST_epoch4_7-30-1-12_prompt_middle", 2),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_promptmiddle_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-30-21-59/epoch4/", "threshold0.5_lagST_epoch4_7-30-21-59_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptmiddle_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-30-21-59/epoch4/", "threshold0.5_lagST_epoch4_7-30-21-59_prompt_short", 2),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_layergate_0.5_lagST-s20.0-lr5e-05-reglr0.05-warmup2/2023-8-7-2-53/epoch2/", "threshold0.5_lagST_epoch2_8-7-2-53_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_layergate_0.5_lagST-s20.0-lr5e-05-reglr0.05-warmup2/2023-8-7-2-53/epoch2/", "threshold0.5_lagST_epoch2_8-7-2-53_eval_prompt_long", 1),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_promptmiddle_layergate_0.5_lagST-s20.0-lr5e-05-reglr0.05-warmup2/2023-7-30-21-29/epoch4/", "spar0.2_threshold0.5_lagST_epoch4_7-30-21-29_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptmiddle_layergate_0.5_lagST-s20.0-lr5e-05-reglr0.05-warmup2/2023-7-30-21-29/epoch4/", "spar0.2_threshold0.5_lagST_epoch4_7-30-21-29_prompt_middle", 2),

    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-22-9/epoch4/", "threshold0.5_lagST_epoch4_7-31-22-9_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-22-9/epoch4/", "threshold0.5_lagST_epoch4_7-31-22-9_prompt_long", 1),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-23/epoch4", "threshold0.5_lagST_epoch4_7-31-21-23_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-7-31-21-23/epoch4", "threshold0.5_lagST_epoch4_7-31-21-23_prompt_long", 1),    
    
    # ("LoRaPruner/gpt4alpaca_llama7b_promptmiddle_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-7-4-19/epoch2/", "threshold0.5_lagST_epoch2_8-7-4-19_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptmiddle_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup2/2023-8-7-4-19/epoch2/", "threshold0.5_lagST_epoch2_8-7-4-19_eval_prompt_middle", 2),

    # ("LoRaPruner/math_llama7b_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup10/2023-7-31-23-4/epoch19/", "math10k_threshold0.5_lagST_epoch19_7-31-23-4_no_prompt", 0),
    # ("LoRaPruner/math_llama7b_layergate_0.5_lagST-s30.0-lr5e-05-reglr0.05-warmup10/2023-7-31-23-4/epoch19/", "math10k_threshold0.5_lagST_epoch19_7-31-23-4_prompt_long", 1),


    # Cubic vs Linear Sparsity
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-7-2-50/epoch3/best/", "8-7-2-50_epoch3_best_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-7-2-50/epoch3/best/", "8-7-2-50_epoch3_best_prompt_long", 1),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-5-2-6/epoch2/", "8-5-2-6_epoch2_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-5-2-6/epoch2/", "8-5-2-6_epoch2_prompt_long", 1),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-5-7-57/epoch3/", "8-5-7-57_epoch3_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-5-7-57/epoch3/", "8-5-7-57_epoch3_prompt_long", 1),
    
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-7-0-17/epoch4/best/", "8-7-0-17_epoch4_best_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-7-0-17/epoch4/best/", "8-7-0-17_epoch4_best_prompt_long", 1),

    # # mark 21
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-13-0-24/epoch6", "8-13-0-24mark20_epoch6_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-13-0-24/epoch6", "8-13-0-24mark20_epoch6_prompt_long", 1),
    
    # # mark 18-1 / or mark 22-1
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-13-0-39/epoch5", "8-13-0-39mark18_22-1_epoch5_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-13-0-39/epoch5", "8-13-0-39mark18_22-1_epoch5_prompt_long", 1),

    # # mark 21-1
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-13-0-39/epoch6", "8-13-0-39mark21-1_epoch6_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-13-0-39/epoch6", "8-13-0-39mark21-1_epoch6_prompt_long", 1),
    
    # # mark 21
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-7-0-17/epoch4", "8-7-0-17_epoch4_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-7-0-17/epoch4", "8-7-0-17_epoch4_prompt_long", 1),
    
    # # mark 22
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-13-0-24/epoch5", "8-13-0-24mark22_epoch5_no_prompt", 0),
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_closeinit_gate2_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-13-0-24/epoch5", "8-13-0-24mark22_epoch5_prompt_long", 1),
    
    ## ================= fix LR scheduler bug ================= ##

    # # mark 24
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-18-46/epoch6", "mark24_epoch6", 1),

    # # mark 25
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-0-8/epoch6", "mark25_epoch6", 1),

    # # mark 26
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-20-28/epoch6", "mark26_epoch6", 2),

    # # mark 27
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-20-20-17/epoch6", "mark27_epoch6", 2),

    # # mark 28
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-20-18-47/epoch6", "mark28_epoch6", 1),

    # # mark 28-1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-21-21-48/epoch6", "mark28-1_epoch6", 1),

    # # mark 29
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-20-8-55/epoch6", "mark29_epoch6", 1),

    # # mark 29-1
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST_CubicSpar-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-21-20-59/epoch6", "mark29-1_epoch6", 1),

    # # mark 50
    # ("LoRaPruner/gptcleaned_llama7b_closeinit_gate_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-22-22-19/epoch6", "mark50_epoch6", 1),

    # # mark 30
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-20-20-31/epoch6", "mark30_epoch6", 2),

    # # mark 31
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-20-20-31/epoch6", "mark31_epoch6", 2),

    # # mark 48
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-21-21-2/epoch5", "mark48_epoch5", 1),

    # # mark 49
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST_CubicSpar-s30.0-lr5e-05-reglr0.05-warmup3/2023-8-21-20-31/epoch5", "mark49_epoch5", 1),
    
    # # mark 32
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s50.0-lr5e-05-reglr0.05-warmup4/2023-8-20-18-46/epoch6", "mark32_epoch6", 1),

    # # mark 33
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s50.0-lr5e-05-reglr0.05-warmup4/2023-8-20-18-46/epoch6", "mark33_epoch6", 1),

    # # mark 34
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate-s50.0-lr5e-05-reglr0.05-warmup4/2023-8-20-20-31/epoch6", "mark34_epoch6", 2),

    # # mark 35
    # ("LoRaPruner/gpt4alpaca_llama7b_closeinit_gate_0.5lagST-s50.0-lr5e-05-reglr0.05-warmup4/2023-8-20-20-31/epoch6", "mark35_epoch6", 2),

    # # mark 51
    # ("LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch8-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-30-2-36/epoch7","mark51_epoch7",2),

    # # mark 52 
    # ("LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch8-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-29-11-2/epoch6","mark52_epoch6",2),

       # # mark 52 
    # ("LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch8-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-29-11-2/epoch4","mark52_epoch4",2),

    # # mark 53
    # ("LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch8-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-29-11-3/epoch7", "mark53_epoch7", 2),

    # # mark 53-1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_epoch8-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-31-4-31/epoch7", "mark53-1_gpt4_epoch7", 2),

    # # mark 54
    # ("LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch9-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-30-2-37/epoch8","mark54_epoch8",2),

    # # mark 55
   # ("LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch9-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-29-21-47/epoch8","mark55_epoch8",2),

    # # mark 56
     #("LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch9-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-29-11-2/epoch8", "mark56_epoch8", 2),

    # # mark 56-1
    # ("LoRaPruner/gptcleaned5k_llama7b_epoch30_warmup10_spar0.2_middle-s20.0-lr5e-05-reglr0.05-warmup10/2023-8-30-0-16/epoch29", "mark56-1_epoch29", 2),
  
    # # mark 57
    # ("LoRaPruner/gptcleaned5k_llama7b_epoch30_warmup10_spar0.2_middle-s20.0-lr5e-05-reglr0.05-warmup10/2023-8-29-5-24/epoch29", "mark57_gpt4_epoch29", 2),

    # # mark 58
    # ("LoRaPruner/gptcleaned5k_llama7b_epoch30_warmup10_spar0.2-s20.0-lr5e-05-reglr0.05-warmup10/2023-8-29-3-30/epoch29", "mark58_gpt4_epoch29", 1),

    # # mark 59
    # LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch8-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-30-2-50

    # # mark 60
    # LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch9-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-30-3-4

    # # mark 61
    # LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch8-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-30-0-16

    # # mark 62
    # LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch9-s30.0-lr5e-05-reglr0.05-warmup4/2023-8-30-0-16

    # # mark 63
    # ("LoRaPruner/gptcleaned5k_llama7b_epoch30_warmup10_spar0.3-s30.0-lr5e-05-reglr0.05-warmup10/2023-8-29-5-26/epoch29", "mark63_gpt4_epoch29", 1),

    # # mark 64
    # LoRaPruner/gptcleaned5k_llama7b_epoch30_warmup10_spar0.3-s30.0-lr5e-05-reglr0.05-warmup10/2023-8-29-5-24

    # # mark 65
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark65-s20.0-lr3e-05-reglr0.05-warmup5/2023-9-2-1-52/epoch7", "mark65_epoch7", 1),

    # # mark 66
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark66-s20.0-lr5e-05-reglr0.05-warmup5/2023-9-1-22-30/epoch7", "mark66_epoch7", 1),

    # # mark 67
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark67-s20.0-lr8e-05-reglr0.05-warmup5/2023-9-2-2-19/epoch7", "mark67_epoch7", 1),

    # # mark 68
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark68-s20.0-lr3e-05-reglr0.05-warmup4/2023-9-1-22-33/epoch7", "mark68_epoch7", 1),

    # # mark 69
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark69-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-1-22-30/epoch7", "mark69_epoch7", 1),

    # # mark 71
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark71-s20.0-lr3e-05-reglr0.05-warmup8/2023-9-2-4-51/epoch14", "mark71_epoch14", 1),

    # # mark 72
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-14-15/epoch14", "mark72_epoch14", "1"),

    # # mark 73
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark73-s20.0-lr8e-05-reglr0.05-warmup8/2023-9-1-22-30/epoch14", "mark73_epoch14", 1),

    # # mark 74
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark74-s20.0-lr3e-05-reglr0.05-warmup10/2023-9-2-3-8/epoch14", "mark74_epoch14", 1),

    # # mark 75
    #  ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark75-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-1-22-30/epoch14", "mark75_epoch14", 1),

    # # mark 76
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark76-s20.0-lr8e-05-reglr0.05-warmup10/2023-9-1-22-29/epoch14", "mark76_epoch14", 1),

    # # mark 77
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark77-s20.0-lr3e-05-reglr0.05-warmup5/2023-9-1-22-30/epoch7", "mark77_epoch7", 1),

    # # mark 78
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark78-s20.0-lr5e-05-reglr0.05-warmup5/2023-9-1-22-59/epoch7", "mark78_epoch7", 1),

    # # mark 79
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark79-s20.0-lr8e-05-reglr0.05-warmup5/2023-9-1-23-0/epoch7", "mark79_epoch7", 1),

    # # mark 80
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark80-s20.0-lr3e-05-reglr0.05-warmup4/2023-9-1-23-33/epoch7", "mark80_epoch7", 1),


    # # mark 81
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark81-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-2-0-47/epoch7", "mark81_epoch7", 1),

    # # mark 82
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark82-s20.0-lr8e-05-reglr0.05-warmup4/2023-9-2-0-40/epoch7", "mark82_epoch7", 1),

    # # mark 83
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark83-s20.0-lr3e-05-reglr0.05-warmup8/2023-9-2-0-47/epoch14", "mark83_epoch14", 1),

    # # mark 84
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark84-s20.0-lr5e-05-reglr0.05-warmup8/2023-9-2-0-48/epoch14", "mark84_epoch14", 1),

    # # mark 85
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark85-s20.0-lr8e-05-reglr0.05-warmup8/2023-9-2-1-8/epoch14", "mark85_epoch14", 1),

    # # mark 86
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark86-s20.0-lr3e-05-reglr0.05-warmup10/2023-9-2-23-36/epoch14", "mark86_epoch14", 1),

    # # mark 87
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark87-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-2-2-20/epoch14", "mark87_epoch14", 1),

    # # mark 88
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark88-s20.0-lr8e-05-reglr0.05-warmup10/2023-9-2-6-37/epoch14", "mark88_epoch14", 1),

    # # mark 89
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark89-s20.0-lr3e-05-reglr0.05-warmup10/2023-9-3-8-29/epoch18", "mark89_epoch18", 1),
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark89-s20.0-lr3e-05-reglr0.05-warmup10/2023-9-3-8-29/epoch19", "mark89_epoch19", 1),

    # # mark 90
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark90-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-3-8-30/epoch16", "mark90_epoch16", 1),
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark90-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-3-8-30/epoch19", "mark90_epoch19", 1),

    # # mark 91
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark91-s20.0-lr8e-05-reglr0.05-warmup10/2023-9-4-2-1

    # # mark 92
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark92-s20.0-lr3e-05-reglr0.05-warmup12/2023-9-3-10-17/epoch16", "mark92_epoch16", 1),
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark92-s20.0-lr3e-05-reglr0.05-warmup12/2023-9-3-10-17/epoch19", "mark92_epoch19", 1),

    # # mark 93
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark93-s20.0-lr5e-05-reglr0.05-warmup12/2023-9-3-11-45/epoch18", "mark93_epoch18", 1),
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark93-s20.0-lr5e-05-reglr0.05-warmup12/2023-9-3-11-45/epoch19", "mark93_epoch19", 1),

    # # mark 94
    #("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark94-s20.0-lr8e-05-reglr0.05-warmup12/2023-9-3-19-23/epoch19","mark94_epoch19", 1),

    # # mark 95
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark95-s20.0-lr3e-05-reglr0.05-warmup15/2023-9-3-9-0/epoch19", "mark95_epoch19", 1),

    # # mark 96
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark96-s20.0-lr5e-05-reglr0.05-warmup15/2023-9-3-9-0/epoch19", "mark96_epoch19", 1),

    # # mark 97
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark97-s20.0-lr8e-05-reglr0.05-warmup15/2023-9-3-9-0/epoch19", "mark97_epoch19", 1),

    # # mark 98
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark98-s20.0-lr3e-05-reglr0.05-warmup10/2023-9-3-9-29/epoch18", "mark98_epoch18", 1),
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark98-s20.0-lr3e-05-reglr0.05-warmup10/2023-9-3-9-29/epoch19", "mark98_epoch19", 1),

    # # mark 99
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark99-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-3-9-32/epoch19", "mark99_epoch19", 1),

    # # mark 100
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark100-s20.0-lr8e-05-reglr0.05-warmup10/2023-9-3-9-44/epoch19", "mark100_epoch19", 1),

    # # mark 101
    # "LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark101-s20.0-lr3e-05-reglr0.05-warmup12/2023-9-3-10-26/epoch10","mark101_epoch19", 1,

    # # mark 102
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark102-s20.0-lr5e-05-reglr0.05-warmup12/2023-9-3-10-30/epoch17", "mark102_epoch17", 1),
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark102-s20.0-lr5e-05-reglr0.05-warmup12/2023-9-3-10-30/epoch19", "mark102_epoch19", 1),

    # # mark 103
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark103-s20.0-lr8e-05-reglr0.05-warmup12/2023-9-3-20-33/epoch19", "mark103_epoch19", 1),
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark103-s20.0-lr8e-05-reglr0.05-warmup12/2023-9-3-20-33/epoch19","mark103_epoch19", 1)

    # # mark 104
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark104-s20.0-lr3e-05-reglr0.05-warmup15/2023-9-3-11-28/epoch19", "mark104_epoch19", 1),

    # # mark 105
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark105-s20.0-lr5e-05-reglr0.05-warmup15/2023-9-3-13-33/epoch19","mark105_epoch19",1),

    # # mark 106
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark106-s20.0-lr8e-05-reglr0.05-warmup15/2023-9-3-14-19/epoch18","mark106_epoch18",1),

     # # mark 107
    # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_mark107-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7","mark107_epoch8",1),

    ##mark 107
   # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_mark107-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch6","mark107_epoch7",1),

    ##mark 108
   # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_cubic_mark108-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7","mark108_epoch8",1),

     ##mark 109
   # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_cubic_mark109-s26.840000000000003-lr5e-05-reglr0.05-warmup5/2023-9-4-2-2/epoch7","mark109_epoch7",1),

    ##mark 109
   # ("LoRaPruner/gpt4alpaca_llama7b_promptlong_cubic_mark109-s26.840000000000003-lr5e-05-reglr0.05-warmup5/2023-9-4-2-2/epoch8","mark109_epoch9",1),

    ##mark 28-2
   # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-4-2-2/epoch7","mark28-2_epoch8",1),

   ##mark 28-4-selected
    #("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-3-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-6-20-50/epoch6","mark28-4_-selected-epoch7",1),

    ##mark 28-4-alllayer
    #("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-4-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-7-1-26/epoch6","mark28-4_-alllayer-epoch7",1),

    ##mark 28-5-selected
    #("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-7-2-21/epoch6","mark28-5_-selected-epoch7",1),

    ##mark 28-5-selected
    #("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-7-2-21/epoch5","mark28-5_-selected-epoch6",1),

     ##mark 28-6 selected
   # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-6-50/epoch6","mark28-6_epoch7",1),

     ##mark 28-6 selected
    #("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-6-50/epoch5","mark28-6_epoch6",1),


    ##mark 28-5-alllayer
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch6","mark28-5_alllayer-epoch7",1),

    ##mark 28-5-alllayer
   # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-6-6-46/epoch5","mark28-5_alllayer-epoch6",1),

     ##mark 28-6-alllayer
    #("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-7-7/epoch3","mark28-6_alllayer-epoch4",1),

    ##mark 28-6-alllayer
  #  ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-6-7-7/epoch6","mark28-6_alllayer-epoch7",1),

    # # mark 110
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark112-s26.840000000000003-lr5e-05-reglr0.05-warmup12/2023-9-4-5-58/epoch19","mark110_epoch19",1),

    # # mark 111
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark111-s26.840000000000003-lr5e-05-reglr0.05-warmup10/2023-9-4-4-15/epoch19","mark111_epoch19",1),

    # # mark 112
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark110-s26.840000000000003-lr5e-05-reglr0.05-warmup10/2023-9-4-4-30/epoch19", "mark112_epoch19", 1),

    # # mark 113
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark113-s26.840000000000003-lr5e-05-reglr0.05-warmup12/2023-9-4-22-25/epoch19","mark113_epoch19",1),

    # # mark 114
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark114-s26.840000000000003-lr5e-05-reglr0.05-warmup14/2023-9-4-4-30/epoch19","mark114_epoch19",1),

    # # mark 115
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark115-s26.840000000000003-lr5e-05-reglr0.05-warmup14/2023-9-5-13-19/epoch19","mark115_epoch19",1),

    # # mark 116
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark116-s26.840000000000003-lr5e-05-reglr0.05-warmup18/2023-9-4-3-13/epoch24","mark116_epoch25",1),

    # # mark 117
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark117-s26.840000000000003-lr5e-05-reglr0.05-warmup18/2023-9-4-3-10/epoch24","mark117_epoch25",1),

    # # mark 118
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark118-s34.56-lr5e-05-reglr0.05-warmup10/2023-9-6-1-37/epoch19","mark118_epoch19",1),

    # # mark 119
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark119-s34.56-lr5e-05-reglr0.05-warmup10/2023-9-5-9-48/epoch19","mark119_epoch19",1),

    # # mark 120
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark120-s34.56-lr5e-05-reglr0.05-warmup12/2023-9-5-11-33/epoch19","mark120_epoch19",1),

    # # mark 121
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark121-s34.56-lr5e-05-reglr0.05-warmup12/2023-9-5-9-50/epoch19","mark121_epoch19",1),

    # # mark 122
    #  ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark122-s34.56-lr5e-05-reglr0.05-warmup14/2023-9-5-11-33/epoch19","mark122_epoch19",1),
    
    # # mark 123
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark123-s34.56-lr5e-05-reglr0.05-warmup14/2023-9-5-10-33/epoch19","mark123_epoch19",1),

    # # mark 125
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark125-s34.56-lr5e-05-reglr0.05-warmup18/2023-9-5-10-47/epoch24", "mark125_epoch24", 1),

    # # mark 125-LLMQAT
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark125-s34.56-lr5e-05-reglr0.05-warmup20/2023-9-9-9-47/epoch24", "mark125_LLM-QAT-epoch24", 1),
    
    
    # # mark 126
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark126-s42.3-lr5e-05-reglr0.05-warmup14/2023-9-5-11-4/epoch19", "mark126_epoch19", 1),
    
    # # mark 127
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark127-s42.3-lr5e-05-reglr0.05-warmup14/2023-9-5-11-4/epoch19", "mark127_epoch19", 1),
    
    # # mark 128
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark128-s42.3-lr5e-05-reglr0.05-warmup18/2023-9-5-11-18/epoch24", "mark128_epoch24", 1),
    
    # # mark 129
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark129-s42.3-lr5e-05-reglr0.05-warmup18/2023-9-5-11-18/epoch24", "mark129_epoch24", 1),


     # # mark 134
   #  ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark134-s26.840000000000003-lr5e-05-reglr0.05-warmup14/2023-9-6-22-59/epoch24", "mark134_epoch25", 1),
     # # mark 135
   #  ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark135-s26.840000000000003-lr5e-05-reglr0.05-warmup16/2023-9-7-18-14/epoch24", "mark135_epoch25", 1),

      # # mark 136
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark136-s34.56-lr5e-05-reglr0.05-warmup14/2023-9-6-23-32/epoch24", "mark136_epoch25", 1),

      # # mark 137
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark135-s26.840000000000003-lr5e-05-reglr0.05-warmup16/2023-9-7-18-14/epoch24", "mark135_epoch25", 1),

      # # mark 138
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark138-s34.56-lr5e-05-reglr0.05-warmup20/2023-9-7-15-53/epoch29", "mark138_epoch30", 1),

      # # mark 139
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark125-s34.56-lr5e-05-reglr0.05-warmup22/2023-9-7-4-21/epoch24", "mark139_epoch25", 1),

     # # mark 130
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark130-s42.3-lr5e-05-reglr0.05-warmup20/2023-9-5-19-40/epoch29", "mark130_epoch30", 1),

     # # mark 131
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark131-s42.3-lr5e-05-reglr0.05-warmup20/2023-9-5-20-59/epoch29", "mark131_epoch30", 1),

     # # mark 132
   #  ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark132-s42.3-lr5e-05-reglr0.05-warmup24/2023-9-5-20-54/epoch29", "mark132_epoch30", 1),

     # # mark 133
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark133-s42.3-lr5e-05-reglr0.05-warmup24/2023-9-6-12-34/epoch29", "mark133_epoch30", 1),

     # # mark 140
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark125-s34.56-lr5e-05-reglr0.05-warmup22/2023-9-7-4-21/epoch24", "mark139_epoch25", 1),
    
    # mark 72-LLMQAT
    #("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark72-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-9-8-47/epoch14", "mark72-LLMQAT_epoch15", 1),

    # mark 84-LLMQAT
    #("LoRaPruner/llmpruner-5k_llama7b_promptlong_mark84-1-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-10-19-15/epoch14", "mark84-LLMQAT_epoch15", 1),

     # mark 84-LLMQAT-promptlong
    #("LoRaPruner/llmpruner-5k_llama7b_promptlong_mark84-1-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-11-19-18/epoch14", "mark84-LLMQAT_-prompt-epoch15", 1),

     # mark 113-LLMQAT
    # ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_cubic_mark113-s26.840000000000003-lr5e-05-reglr0.05-warmup14/2023-9-9-9-59/epoch19", "mark113-LLMQAT_epoch20", 1),

    
    # # mark 28-LLMQAT1 canceled
    # # mark 28-LLMQAT2 canceled
    # # mark 28-LLMQAT3
    # ("LoRaPruner/llmqat32k_mark28LLMQAT3-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-13-23-21/epoch6", "mark28LLMQAT3_epoch7", 1),
    
    # mark 28-LLMQAT4 running
    
    # mark 28-LLMQAT5 done
    # ("LoRaPruner/llmqat32k_mark28LLMQAT5-s26.840000000000003-lr5e-05-reglr0.05-warmup5/2023-9-12-7-32/epoch6", "mark28LLMQAT5_epoch7", 1),
    
    # mark 28-LLMQAT6 done
    #("LoRaPruner/llmqat32k_mark28LLMQAT6-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-11-7-13/epoch6", "mark28LLMQAT6_epoch7", 1),

    # # mark 28-LLMQAT7
    # ("LoRaPruner/llmqat32k_mark28LLMQAT7-s26.840000000000003-lr5e-05-reglr0.05-warmup4/2023-9-13-22-19/epoch6", "mark28LLMQAT7_epoch7", 1),

    # # mark 28-LLMQAT8
    # ("LoRaPruner/llmqat32k_mark28LLMQAT8-s26.840000000000003-lr5e-05-reglr0.05-warmup5/2023-9-13-22-5/epoch6", "mark28LLMQAT8_epoch7", 1),
    
    # mark 28-LLMQAT9 canceled
    
    # mark 28-LLMQAT10 done
    #("LoRaPruner/llmqat32k_mark28LLMQAT10-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-11-7-15/epoch6", "mark28LLMQAT10_epoch7", 1),

    # # mark 28-LLMQAT11
    # ("LoRaPruner/llmqat32k_mark28LLMQAT11-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-13-22-3/epoch6", "mark28LLMQAT11_epoch7", 1),
    
    # # mark 28-LLMQAT12
    # ("LoRaPruner/llmqat32k_mark28LLMQAT12-s34.56-lr5e-05-reglr0.05-warmup5/2023-9-13-22-40/epoch6", "mark28LLMQAT12_epoch7", 1),
    
    
    # mark 28-LLMQAT13 done
    # ("LoRaPruner/llmqat32k_mark28LLMQAT13-s34.56-lr5e-05-reglr0.05-warmup5/2023-9-11-7-40/epoch6", "mark28LLMQAT13_epoch7", 1),

    # mark 28-LLMQAT14 done
    # ("LoRaPruner/llmqat32k_mark28LLMQAT14-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-12-0-9/epoch6", "mark28LLMQAT14_epoch7", 1),
    
    # mark 28-LLMQAT15
   # ("LoRaPruner/llmqat32k_mark28LLMQAT15-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-13-19-13/epoch6", "mark28LLMQAT15_epoch7", 1),
    
    # mark 28-LLMQAT16
   # ("LoRaPruner/llmqat32k_mark28LLMQAT16-s34.56-lr5e-05-reglr0.05-warmup5/2023-9-13-19-13/epoch6", "mark28LLMQAT16_epoch7", 1),

    # # mark 28-LLMQAT21 done
    # ("LoRaPruner/llmqat32k_mark28LLMQAT21-s42.3-lr5e-05-reglr0.05-warmup5/2023-9-11-21-6/epoch6", "mark28LLMQAT21_epoch7", 1),

    # # mark 28-LLMQAT22 done
    # ("LoRaPruner/llmqat32k_mark28LLMQAT22-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-11-19-43/epoch6", "mark28LLMQAT22_epoch7", 1),

    #  # mark 28-LLMQAT23 running

    #  # mark 28-LLMQAT24
   # ("LoRaPruner/llmqat32k_mark28LLMQAT24-s42.3-lr5e-05-reglr0.05-warmup5/2023-9-13-19-14/epoch6", "mark28LLMQAT24_epoch7", 1),
  
    
    
        # mark 25-LLMQAT6-4-3
   # ("LoRaPruner/llmqat-32k_llama7b_mark25LLMQAT32k_4-3-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-11-10-2/epoch6", "mark25LLMQAT4-3_epoch7", 1),

     # mark 25-LLMQAT6-5-2
   # ("LoRaPruner/llmqat-32k_llama7b_mark25LLMQAT32k_4-3-s20.0-lr5e-05-reglr0.05-warmup5/2023-9-11-10-47/epoch6", "mark25LLMQAT5-2_epoch7", 1),

     # mark 25-LLMQAT6-5-2
   # ("LoRaPruner/llmqat-32k_llama7b_mark25LLMQAT32k_4-3-s20.0-lr5e-05-reglr0.05-warmup5/2023-9-11-10-47/epoch5", "mark25LLMQAT5-2_epoch5", 1),


  ##mark 28-5 linear
   # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_LinearSpar_mark28-5_selected-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-13-1-9/epoch6", "mark28-5-linear_epoch7", 1),

   #  # mark 28-6-linear
   # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_LinearSpar_mark28-6_selected-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-13-0-36/epoch6", "mark28-6-linear_epoch7", 1),

   #  # mark 28-6-linear-selected
   # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_LinearSpar_mark28-6-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-13-0-38/epoch5", "mark28-6-linear-selected_epoch6", 1),

   #  # mark 28-6-linear-selected
   #   ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_LinearSpar_mark28-6-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-13-0-38/epoch6", "mark28-6-linear-selected_epoch7", 1),

    # ## mark 25_selectedlayer_96kLLMQAT
    #  ("LoRaPruner/llmqat_llama7b_closeinit_gate_0.5lagST-s20.0-lr5e-05-reglr0.05-warmup5/2023-9-10-0-34/epoch6", "mark25_LLMQAT96k_epoch7", 1),

    #  ## mark 28-6_epoch8
    #  ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_1-4-3-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-18-9-32/epoch5", "mark28-6-8ep_epoch6", 1),
    
    # ## mark 28-6_epoch8
    #  ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_CubicSpar_mark28-6_1-4-3-s42.3-lr5e-05-reglr0.05-warmup4/2023-9-18-9-32/epoch6", "mark28-6-8ep_epoch7", 1),
    
    # c4_ablation_study
    #  ("LoRaPruner/c4_llama7b_wolayer-s20.0-lr5e-05-reglr0.1-warmup10/2023-6-26-10-36/epoch10", "c4_ablation_study", 1),
     
    # ablation study lora apply all
    # ("LoRaPruner/ablation_Study_loraapplyall_v1-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-18-9-32/epoch6/", "Ablation_LoraApplyAll", 1)
    
    # ablation study alpacagpt4 w/o prompt
    # ("LoRaPruner/math_llama7b_wolayer-s20.0-lr5e-05-reglr0.1-warmup2/2023-6-26-20-8/epoch3/", "Ablation_AlpacaGPT4NoPrompt", 1)
    
    
]    

args = []
for ckpt_path, mark, prompt_type in waiting_jobs:
    args.append((ckpt_path, prompt_type, mark))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


