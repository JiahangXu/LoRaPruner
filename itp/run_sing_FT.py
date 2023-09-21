import datetime
import subprocess
import random
import string
import os
import argparse
import torch
from modules.target import target_dict
# from transformers import AutoModelForCausalLM,AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-66b", torch_dtype=torch.float16).cuda()
template = \
"""
description: {job_name}

{target}

code:
  # upload the code
  local_dir: $CONFIG_DIR/../../

storage:
  teamdrive:
    storage_account_name: fastnn
    container_name: teamdrive
    mount_dir: /mnt/data
    local_dir: $CONFIG_DIR/../faketeamdrive/


jobs:
{jobs}
"""


job_eight_nodes = \
"""- name: {job_name}
  sku: G8
  priority: high
 # process_count_per_node: 1
#   execution_mode: managed
  command:
  #- bash setup2.sh
  - bash ./scripts/{file}.sh
  #- python ./itp/sleep.py
  submit_args: 
    env:
      {{DEBUG: 1}}
"""


eval_job_template = \
"""- name: {job_name}
  sku: G{node_num}
  priority: high
 # process_count_per_node: 1
#   execution_mode: managed
  command:
  #- bash setup2.sh
  #- python sleep.py
  - bash ./scripts/{file} {base_model} {ckpt_dir} {prompt_type}
  #- python ./itp/sleep.py
  submit_args: 
    env:
      {{DEBUG: 1}}
"""


eval_harness_job_template = \
"""- name: {job_name}
  sku: G{node_num}
  priority: high
  command:
  - bash ./itp/harness_env.sh
  - bash ./scripts/{file} {base_model} {ckpt_dir} {prompt_type}
  #- python ./itp/sleep.py
  submit_args: 
    env:
      {{DEBUG: 1}}
"""


job_one_node = \
"""- name: {job_name}
  sku: NDAMv4:80G1-A100
  priority: high
  command:
  - bash run_wiki.sh
  submit_args: 
    env:
      {{DEBUG: 1}}
"""
# --nproc_per_node=8 --node_rank=$${{NODE_RANK}} --nnodes=4 
#     --master_addr=$${{MASTER_ADDR}} --master_port=$${{MASTER_PORT}}  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='sing_octo', choices=list(target_dict.keys()), help='where to submit')
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--sparsity", type=float, required=False)
    parser.add_argument("--file", type=str, required=False)
    parser.add_argument("--task_name", type=str, required=False, choices=[
    "mnli",
    "qqp",
    "qnli",
    "sst2",
    "stsb",
    "mrpc",
    "squad",
    "squad_v2",
    "wikitext2",
    "wikitext2_eval",
    "alpaca",
    "gpt4alpaca",
    "alpacaclean",
    "c4",
    "piqa",
    "math10k",
    "storycloze",
    "arc-e",
    "arc-c",
    "gsm8k",
    "multiarith",
    "boolqa",
    "hellaswag",
    "obqa",
    "winogrande",
    "open_orca",
    "harness",
    "nqopen",
    "reasoning",
    "triviaqa",
    "race",
    "squad",
    "race_high",
    "race_middle"
    ])
    parser.add_argument("--base_model", type=str, required=False)
    parser.add_argument("--ckpt_dir", type=str, required=False)
    parser.add_argument("--prompt_type", type=str, required=False)
    parser.add_argument("--mark", type=str, required=False)
    parser.add_argument("--node_num", type=int, default=2, required=False)
    
    # parser.add_argument("--constraint", type=float, required=True,
    # help="MAC/latency constraint relative to the original model",
    # )
    args = parser.parse_args()

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0

    if args.task_name in ["wikitext2_eval", "piqa", "math_eval", "piqa", "storycloze", "arc-e", "arc-c", \
      "math_eval", "boolqa", "hellaswag", "obqa", "winogrande", "gsm8k", "multiarith", "c4"]:
        job_template = eval_job_template
        date = datetime.datetime.now().strftime('%m%d%H%M')
        job_name = f'{args.model_name.replace("%","")}-{args.task_name}-{args.mark}-{date}'
        jobs = job_template.format(
            job_name=job_name, 
            debug=mode,
            file = args.file,
            ckpt_dir = args.ckpt_dir,
            prompt_type = args.prompt_type,
            node_num = args.node_num,
            base_model = args.base_model,
        )
    elif args.task_name in ["harness", "nqopen", "reasoning", "triviaqa", "squad", "race", "race_high", "race_middle"]:
        job_template = eval_harness_job_template
        date = datetime.datetime.now().strftime('%m%d%H%M')
        job_name = f'{args.model_name.replace("%","")}-{args.task_name}-{args.mark}-{date}'
        jobs = job_template.format(
            job_name=job_name, 
            debug=mode,
            file = args.file,
            ckpt_dir = args.ckpt_dir,
            prompt_type = args.prompt_type,
            node_num = args.node_num,
            base_model = args.base_model,
        )
    else:
        job_template = job_eight_nodes
        date = datetime.datetime.now().strftime('%m%d%H%M')
        job_name = f'{args.model_name.replace("%","")}-{args.task_name}-{args.sparsity}-{date}'
        jobs = job_template.format(
            job_name=job_name, 
            debug=mode,
            file = args.file
        )
    description = f'{job_name}'

    # ======================================================================================================
    # Don't need to modify following code
    result = template.format(
        job_name=job_name,
        jobs=jobs,
        target=target_dict[args.target], 
    )   
    print(result)

    tmp_name = ''.join(random.choices(string.ascii_lowercase, k=6)) + job_name
    tmp_name = os.path.join("./.tmp", tmp_name)
    with open(tmp_name, "w") as fout:
        fout.write(result)
    if mode == 0:
        subprocess.run(["amlt", "run", "-t", "local", "--use-sudo", tmp_name, "--devices", "all"])
    else:
        # subprocess.run(f'amlt run -d {description} {tmp_name} {job_name}', shell=True)
        subprocess.run(["amlt", "run", "-d", description, tmp_name, job_name])

   
if __name__ == "__main__":
    main()