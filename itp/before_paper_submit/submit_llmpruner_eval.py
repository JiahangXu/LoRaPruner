import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path = arg
    mrk = ckpt_path
    prompt_type = 0
    if prompt_type == 0:
        mark = mrk + f"_no_prompt"
    else:
        mark = mrk + f"_prompt_type{prompt_type}"
    command_head = "python run_sing.py submit --target sing_research --model_name eval_llmpruner "
    command_list_no_prompt = []
    command_list = [
        command_head + f"--file evaluation_llmpruner/eval_nqopen.sh --task_name nqopen --ckpt_dir {ckpt_path} --mark {mark} --node_num 1",
        command_head + f"--file evaluation_llmpruner/eval_triviaqa.sh --task_name triviaqa --ckpt_dir {ckpt_path} --mark {mark} --node_num 1",
        command_head + f"--file evaluation_llmpruner/eval_race.sh --task_name race --ckpt_dir {ckpt_path} --mark {mark} --node_num 1",
        command_head + f"--file evaluation_llmpruner/eval_squad.sh --task_name squad --ckpt_dir {ckpt_path} --mark {mark} --node_num 1",
    ]
    if prompt_type == 0:
        command_list = command_list_no_prompt + command_list
    for command in command_list:
        os.system(command)


waiting_jobs = ["5.4b"] # , "5b", "4.8b", "4.5b", "4b", "3.5b"]    

args = []
for ckpt_path in waiting_jobs:
    args.append((ckpt_path))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


