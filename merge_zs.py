import os
import argparse
from models.modeling_llama import LlamaForCausalLM
from utils.cofi_utils import load_zs
from models.modeling_llama import LlamaConfig
import torch

def update_params(lm_model, zs):
    model = lm_model.model

    config = lm_model.config
    hidden_dims = config.hidden_size
    num_heads = config.num_attention_heads
    dims_per_head = hidden_dims // num_heads
    num_layers = config.num_hidden_layers
    if zs is not None:
        if "intermediate_z" in zs:
            for layer in range(num_layers):
                if "mlp_z" in zs and zs["mlp_z"][layer] == 0:
                    continue
                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()
                model.layers[layer].mlp.gate_proj.weight.data = model.layers[layer].mlp.gate_proj.weight.transpose(0, 1).data.mul(intermediate_z).transpose(0, 1)
                model.layers[layer].mlp.up_proj.weight.data = model.layers[layer].mlp.up_proj.weight.transpose(0, 1).data.mul(intermediate_z).transpose(0, 1)
                
        if "head_z" in zs:
            for layer in range(num_layers):
                if "head_layer_z" in zs and zs["head_layer_z"][layer] == 0:
                    continue
                head_z = zs["head_z"][layer].cpu().squeeze().clone()
                head_z = torch.repeat_interleave(head_z, dims_per_head)
                model.layers[layer].self_attn.v_proj.weight.data = model.layers[layer].self_attn.v_proj.weight.transpose(0, 1).data.mul(head_z).transpose(0, 1)

        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"].cpu().squeeze().clone()
            for layer in range(num_layers):
                model.layers[layer].self_attn.o_proj.weight.data = model.layers[layer].self_attn.o_proj.weight.transpose(0, 1).data.mul(hidden_z).transpose(0, 1)
                model.layers[layer].mlp.down_proj.weight.data = model.layers[layer].mlp.down_proj.weight.transpose(0, 1).data.mul(hidden_z).transpose(0, 1)


def main():
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')
    parser.add_argument('--model_name_or_path', type=str, help='base model name')
    parser.add_argument('--pretrained_pruned_model', type=str, help='pretrained pruned model name')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    # model initialize
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, config=config)

    zs = load_zs(os.path.join(args.pretrained_pruned_model, 'zs.pt'))
    for key in zs:
        zs[key] = zs[key].detach()
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

    update_params(model, zs)
    
    output_path = "./llama_merged" if args.output_dir == None else args.output_dir
    model.save_pretrained(output_path)

if __name__ == "__main__":
    main()
