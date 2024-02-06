import datetime
import subprocess
import random
import string
import os
import argparse
import torch
from modules.target import target_dict

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

job_template = \
"""- name: {job_name}
  sku: G1
  priority: high
  command:
  - pip install transformers==4.30.1
  - python generate_data.py {chunk}
  submit_args: 
    env:
      {{DEBUG: 1}}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='sing_octo', choices=list(target_dict.keys()), help='where to submit')
    parser.add_argument("--chunk", type=int, default=0, required=False)
    
    # parser.add_argument("--constraint", type=float, required=True,
    # help="MAC/latency constraint relative to the original model",
    # )
    args = parser.parse_args()

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0

    job_name = f"Generate_LLM-QAT_test_turn1lenght35_chunk{args.chunk}"
    jobs = job_template.format(
        job_name=job_name, 
        debug=mode,
        chunk=args.chunk
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