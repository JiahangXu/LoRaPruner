import os
import argparse
from peft import PeftModel
import torch
from models.modeling_llama import LlamaForCausalLM

refered_files_path = "./llama_pruned_a100_5.4B"

def main():
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')
    parser.add_argument('--model_name_or_path', type=str, help='base model name')
    parser.add_argument('--peft_path', type=str, help='pretrained pruned model name')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    llama = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
    output_path = "./llama_pruned" if args.output_dir == None else args.output_dir
    if args.peft_path is not None:
        model = PeftModel.from_pretrained(llama, args.peft_path, torch_dtype=torch.float16)

    merged_llama = model.merge_and_unload()
    merged_llama.save_pretrained(output_path)
    os.system(f"cp {refered_files_path}/tokenizer.model {output_path}")
    os.system(f"cp {refered_files_path}/tokenizer.json {output_path}")
    os.system(f"cp {refered_files_path}/tokenizer_config.json {output_path}")
    os.system(f"cp {refered_files_path}/special_tokens_map.json {output_path}")
    print(f"Save merged Checkpoint! (Output path: {output_path})")
    
if __name__ == "__main__":
    main()
