import os
from tqdm.contrib.concurrent import process_map

def single_job(arg):
    ckpt_path, pt, mrk, lora_param = arg
    for prompt_type in [0, "1-1", pt]:
        if prompt_type == 0:
            mark = mrk + f"_no_prompt"
        else:
            mark = mrk + f"_prompt_type{prompt_type}"
        command_head = "python run_sing.py submit --target sing_octo --model_name eval_llama2-7b "
        command_list = [
            command_head + f"--file evaluation_llama2/eval_harness.sh --task_name harness --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --lora_param {lora_param} --node_num 1",
            # command_head + f"--file evaluation_llama2/eval_wikitext.sh --task_name wikitext2_eval --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --lora_param {lora_param} --node_num 1",
            # command_head + f"--file evaluation/eval_mmlu.sh --task_name mmlu --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
            # command_head + f"--file evaluation/eval_bbh.sh --task_name bbh --ckpt_dir /mnt/data/{ckpt_path} --prompt_type {prompt_type} --mark {mark} --node_num 1",
        ]
        for command in command_list:
            os.system(command)

waiting_jobs = [
    ("LoRaPruner/llama2-7b_mark25-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-29-5-27/epoch6", "llama2-7b_mark25_epoch6", 1, "Q.V"),
    ("LoRaPruner/llama7b-2_mark25_LoRaApplyAll-s20.0-lr5e-05-reglr0.05-warmup4/2023-9-29-7-14/epoch6", "llama2-7b_mark25_LoRaApplyAll_epoch6", 1, "Q.K.V.O.F"),
    
    ("LoRaPruner/llama2-7b_mark28-2-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-29-22-15/epoch7", "llama2-7b_mark28-2_epoch6", 1, "Q.V"),
    # (),
    ("LoRaPruner/llama2-7b_mark28-4-s26.83-lr5e-05-reglr0.05-warmup4/2023-9-29-22-16/epoch6", "llama2-7b_mark28-4_epoch6", 1, "Q.V"),
    # ("LoRaPruner/llama7b-2_mark28-4_LoRaApplyAll-s26.83-lr5e-05-reglr0.05-warmup4/2023-10-4-9-28"),
    
    ("LoRaPruner/llama2-7b_mark28-5-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-29-22-16/epoch6", "llama2-7b_mark28-5_epoch6", 1, "Q.V"),
    ("LoRaPruner/llama7b-2_mark28-5_LoRaApplyAll-s34.56-lr5e-05-reglr0.05-warmup4/2023-9-29-22-16/epoch6", "llama2-7b_mark28-5_LoRaApplyAll_epoch6", 1, "Q.K.V.O.F"),
    
    # ("LoRaPruner/llama2-7b_mark28-6-s42.3-lr5e-05-reglr0.05-warmup4/2023-10-5-7-18"),
    # ("LoRaPruner/llama7b-2_mark28-6_LoRaApplyAll-s42.3-lr5e-05-reglr0.05-warmup4/2023-10-5-7-18")
      

]

args = []
for ckpt_path, mark, prompt_type, lora_param in waiting_jobs:
    args.append((ckpt_path, prompt_type, mark, lora_param))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")


