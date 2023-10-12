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
            config.lora_layers = config.num_hidden_layers
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
            if not args.lora_merged:
                lora_ckpt = os.path.join(args.lora_ckpt, 'lora_weights.pt')
            from models.l0_module import L0Module
            l0_module = L0Module(config=_config)
            zs = torch.load(os.path.join(args.lora_ckpt, 'zs.pt'), map_location="cpu")

            if zs["head_z"].shape[0] < _config.num_hidden_layers:
                if zs["head_z"].shape[0] == 26:
                    zs["head_z"] = torch.concat([torch.ones(4, 1, 32, 1, 1), zs["head_z"], torch.ones(2, 1, 32, 1, 1)])
                    zs["intermediate_z"] = torch.concat([torch.ones(4, 1, 1, 11008), zs["intermediate_z"], torch.ones(2, 1, 1, 11008)])
                elif zs["head_z"].shape[0] == 28:
                    zs["head_z"] = torch.concat([torch.ones(3, 1, 32, 1, 1), zs["head_z"], torch.ones(1, 1, 32, 1, 1)])
                    zs["intermediate_z"] = torch.concat([torch.ones(3, 1, 1, 11008), zs["intermediate_z"], torch.ones(1, 1, 1, 11008)])

            if "layer_z" in zs:
                zs['head_layer_z'] = zs['layer_z']
                zs['mlp_z'] = zs['layer_z']
                zs.pop('layer_z')
            for key in zs:
                zs[key] = zs[key].cuda().detach().half()
            zs = zs

            sparsity_info = l0_module.calculate_model_size(zs)

            if "decapoda-research/llama-13b-hf" in args.base_model:
                embedding_parmas = 334648320
                model_params = 13022417920
            else:
                embedding_parmas = 262410240
                model_params = 6738415616

            print(f"Model path: {args.lora_ckpt}")
            print(f"Sparsity: {sparsity_info['pruned_model_sparsity']}")
            print(f"Ramaining Params: {sparsity_info['remaining_params'] + embedding_parmas}")
            print(f"Sparsity: {1 - (sparsity_info['remaining_params'] + embedding_parmas) / model_params}")

        model = LlamaForCausalLM.from_pretrained(
            LlamaForCausalLM,
            args.base_model,
            from_tf=False,
            config=_config,
            use_auth_token="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",
            lora_ckpt = lora_ckpt
        )
        description = "Model Type: {}\n LoRaPruner Model: {}\n LORA ckpt: {}".format(args.model_type, args.base_model, args.lora_ckpt)

    elif args.model_type == 'llm_pruner':
        try:
            from peft import PeftModel
        except:
            raise NotImplementedError
        config = LlamaConfig.from_pretrained(
            args.ckpt,
            # trust_remote_code=trust_remote_code,
            # revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )
        config.use_cache = False
        lora_ckpt = None
        config.use_lora = False

        tokenizer = LlamaTokenizer.from_pretrained(
            args.ckpt,
            use_auth_token="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",
            padding_side="left",
            truncation_side="left",
        )
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.model_max_length = max_length

        
        zs = {}
        if args.lora_ckpt is not None:
            if not args.lora_merged:
                lora_ckpt = os.path.join(args.lora_ckpt, 'lora_weights.pt')
            from models.l0_module import L0Module
            l0_module = L0Module(config=config)
            zs = torch.load(os.path.join(args.lora_ckpt, 'zs.pt'), map_location="cpu")

            if zs["head_z"].shape[0] < config.num_hidden_layers:
                if zs["head_z"].shape[0] == 26:
                    zs["head_z"] = torch.concat([torch.ones(4, 1, 32, 1, 1), zs["head_z"], torch.ones(2, 1, 32, 1, 1)])
                    zs["intermediate_z"] = torch.concat([torch.ones(4, 1, 1, 11008), zs["intermediate_z"], torch.ones(2, 1, 1, 11008)])
                elif zs["head_z"].shape[0] == 28:
                    zs["head_z"] = torch.concat([torch.ones(3, 1, 32, 1, 1), zs["head_z"], torch.ones(1, 1, 32, 1, 1)])
                    zs["intermediate_z"] = torch.concat([torch.ones(3, 1, 1, 11008), zs["intermediate_z"], torch.ones(1, 1, 1, 11008)])

            if "layer_z" in zs:
                zs['head_layer_z'] = zs['layer_z']
                zs['mlp_z'] = zs['layer_z']
                zs.pop('layer_z')
            for key in zs:
                zs[key] = zs[key].cuda().detach().half()
            zs = zs

            sparsity_info = l0_module.calculate_model_size(zs)

            if "decapoda-research/llama-13b-hf" in args.base_model:
                embedding_parmas = 334648320
                model_params = 13022417920
            else:
                embedding_parmas = 262410240
                model_params = 6738415616

            print(f"Model path: {args.lora_ckpt}")
            print(f"Sparsity: {sparsity_info['pruned_model_sparsity']}")
            print(f"Ramaining Params: {sparsity_info['remaining_params'] + embedding_parmas}")
            print(f"Sparsity: {1 - (sparsity_info['remaining_params'] + embedding_parmas) / model_params}")

        model = LlamaForCausalLM.from_pretrained(
                LlamaForCausalLM,
                args.ckpt,
                from_tf=False,
                config=config,
                use_auth_token="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",
                lora_ckpt = lora_ckpt
            )
        
        if args.lora_ckpt is not None:
            model = PeftModel.from_pretrained(
                model,
                args.lora_ckpt,
                torch_dtype=torch.float16,
            )
        
        description = "Model Type: {} Finetuned LoRa-Pruner in LLM-Pruner way.\n Model: {}\n LORA ckpt: {}".format(args.model_type, args.ckpt, args.lora_ckpt)

    else:
        raise NotImplementedError
    print(description)
    
    if args.device != "cpu":
        model.half()
    model.to(args.device)

    model.eval()
    if args.eval_c4:
        ppl = PPLMetric(model, tokenizer, ['c4'], args.max_seq_len, device=args.device, batch_size=1, zs=zs, prompt_mark=args.prompt_mark)
    else:
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.device, batch_size=1, zs=zs, prompt_mark=args.prompt_mark)
    print("Prompt mark: {}; PPL: {}".format(args.prompt_mark, ppl))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--model_type', type=str, default="pretrain")
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--prompt_mark', type=str, default="0", help='prompt mark')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--eval_c4', action="store_true")
    parser.add_argument('--lora_merged', action="store_true")
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
