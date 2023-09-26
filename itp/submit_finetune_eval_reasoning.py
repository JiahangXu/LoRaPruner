import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path, pt, mrk = arg
    for prompt_type in [0, "1-1", pt]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing.py submit --target sing_octo --model_name eval_Finetuned_llama7b "
        command_list_no_prompt = [
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_boolqa.sh --task_name boolqa --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 2",
            # command_head + f"--file evaluation_finetune/zeroshot/eval_llama7b_hellaswag.sh --task_name hellaswag --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 8",
        ]
        command_list = [
            # command_head + f"--file evaluation_finetune/eval_harness.sh --task_name harness --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_nqopen.sh --task_name nqopen --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_triviaqa.sh --task_name triviaqa --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_reasoning.sh --task_name reasoning --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_squad.sh --task_name squad --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation_merged/eval_mmlu.sh --task_name mmlu --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation_merged/eval_bbh.sh --task_name bbh --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            command_head + f"--file evaluation_merged/eval_race_high.sh --task_name race_high --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_race_middle.sh --task_name race_middle --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_wikitext.sh --task_name wikitext2_eval --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation_finetune/eval_c4.sh --task_name c4 --base_model /mnt/data/{base_model} --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
        ]
        if prompt_type == 0:
            command_list = command_list_no_prompt + command_list
        for command in command_list:
            os.system(command)


waiting_jobs = [
    # ("LoRaPruner/best_candidate_with_zs_merged/mark25_ppl14.95", "mark25", 1),
    
    # ("LoRaPruner/best_candidate_with_zs_merged/mark28-2_ppl20.20", "mark28-2", 1),
    
    # ("LoRaPruner/best_candidate_with_zs_merged/mark28-5_ppl23.52", "mark28-5", 1),
    
    # ("LoRaPruner/best_candidate_with_zs_merged/mark28-6_ppl41.69", "mark28-6", 1),
    
    # # # Finetune mark25FT9
    # ("LoRaPruner/best_candidate_with_zs_merged/mark25FT9_ppl14.58", "mark25FT9", 1),
    
    # # # Finetune mark25FT9EvalLong
    ("LoRaPruner/best_candidate_with_zs_merged/mark25FT9EvalLong_ppl14.58", "mark25FT9EvalLong", 1),
    
    # # # Finetune mark25FT12
    ("LoRaPruner/best_candidate_with_zs_merged/mark25FT12_ppl14.35", "mark25FT12", 1),

    # # Finetune mark25FT15
    # ("LoRaPruner/best_candidate_with_zs_merged/mark25FT15_ppl14.66", "mark25FT15_epoch0", 1),
    
    # # Finetune mark28-2_1.6e-3-epoch1
    # ("LoRaPruner/best_candidate_with_zs_merged/mark28-2_1.6e-3_ppl19.54", "mark28-2_1.6e-3_epoch1", 1),

    # # Finetune mark28-5_4e-4-epoch1
    # ("LoRaPruner/best_candidate_with_zs_merged/mark28-5_4e-4_ppl22.32", "mark28-5_4e-4_epoch1", 1),

    # ablation study c4
    #  ("LoRaPruner/best_candidate_with_zs_merged/mark_c4ablation_ppl15.61", "c4_ablation_study", 1),
    
    # # ablation study noprompt
    # ("LoRaPruner/best_candidate_with_zs_merged/markNoPromptAblation_ppl18.76", "Ablation_AlpacaGPT4NoPrompt", 1)
    
    # mark 84 LLMQAT
    # ("LoRaPruner/best_candidate_with_zs_merged/mark84LLMQAT_ppl13.12", "mark84LLMQAT_5e-5", 1)

]

args = []
for base_model, mark, prompt_type in waiting_jobs:
    args.append((base_model, prompt_type, mark))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


