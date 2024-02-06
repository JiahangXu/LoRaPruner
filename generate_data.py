# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
from tqdm import tqdm

print("Loading tokenizer")
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
print("Tokenizer loaded!")
print("Loading model")
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
print("Model loaded!")

model = model.half()
model = model.cuda()
model = model.eval()

os.system("mkdir gen_data")

n_vocab = 500 # number of initial tokens for synthesizing data on each GPU.

i_start = sys.argv[1]
if os.path.exists("gen_data/gen.chunk35."+str(i_start).zfill(2)+".jsonl"):
    with open("gen_data/gen.chunk35."+str(i_start).zfill(2)+".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

for j in [3, 5]:
    for i in tqdm(range(int(i_start) * n_vocab + inner_loop, (int(i_start)+1) * n_vocab)):
        print(i)
        input_ids = torch.tensor([[i]]).cuda()
        print("generating")
        outputs1 = model.generate(input_ids, do_sample=False, max_length=j)
        outputs = model.generate(outputs1, do_sample=True, max_length=1024)
        
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        text_dict = {"text" : gen_text[0]}
        with open("gen_data/gen.chunk35."+str(i_start).zfill(2)+".jsonl", "a") as f:
            f.write(json.dumps(text_dict))
            f.write('\n')

os.system("cp gen_data/* /mnt/data/LoRaPruner/LLM-QAT_gen_data/")
