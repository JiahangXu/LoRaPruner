import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig

from models.modeling_llama import LlamaForCausalLM
from models.tokenization_llama import LlamaTokenizer
from models.modeling_llama import LlamaConfig
from ppl_new import PPLMetric

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    set_random_seed(args.seed)

    zs = {}
    if args.model_type == 'lora_pruner':

        def set_lora_args(config):
            config.use_lora = True
            config.lora_rank = 8
            config.lora_train_bias = None
            config.lora_alpha = 8.0
            config.lora_param = "Q.V"
            config.lora_layers = 32
            return config
        
        _config = LlamaConfig.from_pretrained(
            args.base_model,
        )
        _config.use_cache = False
        lora_ckpt = None
        _config = set_lora_args(_config)
        
        tokenizer = LlamaTokenizer.from_pretrained(
            args.base_model,
            use_auth_token="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",
            padding_side="left",
            truncation_side="left",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 1024

        if args.lora_ckpt is not None:
            lora_ckpt = os.path.join(args.lora_ckpt, 'lora_weights.pt')
            l0_module = torch.load(os.path.join(args.lora_ckpt, 'l0_module.pt'), map_location="cpu")
            zs = l0_module.forward(training=False)
            if "layer_z" in zs:
                zs['head_layer_z'] = zs['layer_z']
                zs['mlp_z'] = zs['layer_z']
                zs.pop('layer_z')
            for key in zs:
                zs[key] = zs[key].cuda().detach().half()
            zs = zs

        model = LlamaForCausalLM.from_pretrained(
            LlamaForCausalLM,
            args.base_model,
            from_tf=False,
            config=_config,
            use_auth_token="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",
            lora_ckpt = lora_ckpt
        )
        description = "Model Type: {}\n LoRaPruner Model: {}\n LORA ckpt: {}".format(args.model_type, args.base_model, args.lora_ckpt)

    else:
        raise NotImplementedError
    print(description)
    
    if args.device != "cpu":
        model.half()
    model.to(args.device)

    model.eval()
    if args.max_seq_len == None:
        for max_seq_len in [128, 1024]:
            ppl = PPLMetric(model, tokenizer, ['wikitext2'], max_seq_len, device=args.device, batch_size=1, zs=zs)
            print("Seq_len: {}; PPL before pruning: {}".format(max_seq_len, ppl))
    else:
        ppl = PPLMetric(model, tokenizer, ['wikitext2'], args.max_seq_len, device=args.device, batch_size=1, zs=zs)
        print("Seq_len: {}; PPL before pruning: {}".format(args.max_seq_len, ppl))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--model_type', type=str, default="pretrain")
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
