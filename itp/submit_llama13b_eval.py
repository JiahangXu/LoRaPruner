import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path, pt, mrk, lora_param = arg
    for prompt_type in [0]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing.py submit --target sing_octo --model_name eval_llama13b "
        command_list = [
            # command_head + f"--file evaluation_llama13b/eval_harness.sh --task_name harness --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --lora_param {lora_param} --mark {mark} --node_num 4",
            # command_head + f"--file evaluation_llama13b/eval_wikitext.sh --task_name wikitext2_eval --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --lora_param {lora_param} --mark {mark} --node_num 4",
            command_head + f"--file evaluation_llama13b/eval_mmlu.sh --task_name mmlu --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
            command_head + f"--file evaluation_llama13b/eval_bbh.sh --task_name bbh --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 4",
        ]
        for command in command_list:
            os.system(command)

waiting_jobs = [
    ("LoRaPruner/llama13b_mark25_1-4-2-s15.939999999999998-lr5e-05-reglr0.05-warmup4/2023-9-30-12-58/epoch6", "llama13b_mark25_1-4-2_epoch6", 1, "Q.V"),
    # (),
    # ("LoRaPruner/llama13b_mark25_1-4-3-s15.939999999999998-lr5e-05-reglr0.05-warmup4/2023-9-30-19-33/epoch7"),
    # ("LoRaPruner/llama13b_mark25_1-4-3_LoRaApplyAll-s15.939999999999998-lr5e-05-reglr0.05-warmup4/2023-9-30-13-31/epoch7")
    
    # ("LoRaPruner/llama13b_mark28-2_1-4-3-s23.82-lr5e-05-reglr0.05-warmup4/2023-9-30-20-0/epoch7"),
    # ("LoRaPruner/llama13b_mark28-2_1-4-3_LoRaApplyAll-s23.82-lr5e-05-reglr0.05-warmup4/2023-9-30-20-1/epoch7"),
    # (),
    # ("LoRaPruner/llama13b_mark28-4_1-4-2_LoRaApplyAll-s23.82-lr5e-05-reglr0.05-warmup4/2023-9-30-20-51/epoch6"),
    
    # ("LoRaPruner/llama13b_mark28-5_1-4-3-s31.7-lr5e-05-reglr0.05-warmup4/2023-9-30-20-1/epoch7"),
    # (),
    # (),
    # (),

]

args = []
for ckpt_path, mark, prompt_type, lora_param in waiting_jobs:
    args.append((ckpt_path, prompt_type, mark, lora_param))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


