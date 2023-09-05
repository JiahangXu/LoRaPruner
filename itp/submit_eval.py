import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path, pt, mrk = arg
    for prompt_type in [0, "1-1", pt]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing.py submit --target sing_research --model_name eval_llama7b "
        command_list_no_prompt = [
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
        ]
        command_list = [
            command_head + f"--file evaluation/eval_harness.sh --task_name harness --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation/eval_wikitext.sh --task_name wikitext2_eval --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation/eval_c4.sh --task_name c4 --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
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
    # LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch8-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-30-2-36

    # # mark 52 
    # LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch8-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-29-11-2

    # # mark 53
    # ("LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch8-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-29-11-3/epoch7", "mark53_epoch7", 2),

    # # mark 53-1
    # ("LoRaPruner/gpt4alpaca_llama7b_prompt_nogate_epoch8-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-31-4-31/epoch7", "mark53-1_gpt4_epoch7", 2),

    # # mark 54
    # LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch9-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-30-2-37

    # # mark 55
    # LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch9-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-29-21-47

    # # mark 56
    # ("LoRaPruner/alpacacleaned_llama7b_prompt_nogate_epoch9-s20.0-lr5e-05-reglr0.05-warmup4/2023-8-29-11-2/epoch8", "mark56_epoch8", 2),

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
    ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark92-s20.0-lr3e-05-reglr0.05-warmup12/2023-9-3-10-17/epoch16", "mark92_epoch16", 1),
    ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark92-s20.0-lr3e-05-reglr0.05-warmup12/2023-9-3-10-17/epoch19", "mark92_epoch16", 1),

    # # mark 93
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark93-s20.0-lr5e-05-reglr0.05-warmup12/2023-9-3-11-45

    # # mark 94
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark94-s20.0-lr8e-05-reglr0.05-warmup12/2023-9-3-19-23

    # # mark 95
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark95-s20.0-lr3e-05-reglr0.05-warmup15/2023-9-3-9-0

    # # mark 96
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark96-s20.0-lr5e-05-reglr0.05-warmup15/2023-9-3-9-0

    # # mark 97
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark97-s20.0-lr8e-05-reglr0.05-warmup15/2023-9-3-9-0

    # # mark 98
    ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark98-s20.0-lr3e-05-reglr0.05-warmup10/2023-9-3-9-29/epoch18", "mark98_epoch16", 1),
    ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark98-s20.0-lr3e-05-reglr0.05-warmup10/2023-9-3-9-29/epoch19", "mark98_epoch16", 1),

    # # mark 99
    ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark99-s20.0-lr5e-05-reglr0.05-warmup10/2023-9-3-9-32/epoch19", "mark99_epoch16", 1),

    # # mark 100
    ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark100-s20.0-lr8e-05-reglr0.05-warmup10/2023-9-3-9-44/epoch19", "mark100_epoch16", 1),

    # # mark 101
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark101-s20.0-lr3e-05-reglr0.05-warmup12/2023-9-3-10-26

    # # mark 102
    ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark102-s20.0-lr5e-05-reglr0.05-warmup12/2023-9-3-10-30/epoch17", "mark102_epoch16", 1),
    ("LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark102-s20.0-lr5e-05-reglr0.05-warmup12/2023-9-3-10-30/epoch19", "mark102_epoch16", 1),

    # # mark 103
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark103-s20.0-lr8e-05-reglr0.05-warmup12/2023-9-3-20-33

    # # mark 104
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark104-s20.0-lr3e-05-reglr0.05-warmup15/2023-9-3-11-28

    # # mark 105
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark105-s20.0-lr5e-05-reglr0.05-warmup15/2023-9-3-13-33

    # # mark 106
    # LoRaPruner/gpt4alpaca-5k_llama7b_promptlong_mark106-s20.0-lr8e-05-reglr0.05-warmup15/2023-9-3-14-19
]    

args = []
for ckpt_path, mark, prompt_type in waiting_jobs:
    args.append((ckpt_path, prompt_type, mark))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


