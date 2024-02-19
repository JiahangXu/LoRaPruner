import os
import sys
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from args import AdditionalArguments, DataTrainingArguments
from models.modeling_llama import LlamaForCausalLM
from models.modeling_llama import LlamaConfig
from models.model_args import ModelArguments

refered_files_path = "./llama_pruned_a100_5.4B"

def set_lora_args(config, modeling_args):
    config.use_lora = modeling_args.use_lora
    config.lora_rank = modeling_args.lora_rank
    config.lora_train_bias = modeling_args.lora_train_bias
    config.lora_alpha = modeling_args.lora_alpha
    config.lora_param = modeling_args.lora_param
    config.lora_layers = modeling_args.lora_layers
    return config


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    # model initialize
    if model_args.training_objective == "LM":
        config = LlamaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            #num_labels=num_labels,
            #finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True,
        )
        config.use_cache = False
        lora_ckpt = None
        config = set_lora_args(config, model_args)
        if additional_args.pretrained_pruned_model is not None:
            lora_ckpt = os.path.join(additional_args.pretrained_pruned_model, 'lora_weights.pt')
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            lora_ckpt = lora_ckpt
        )
    else:
        raise ValueError("Training objective should be either cls or clm")
    
    config.use_lora = False
    llama = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    output_path = "./llama_pruned" if training_args.output_dir == "./" else training_args.output_dir
    llama.load_state_dict(model.state_dict(), strict=False)
    llama.save_pretrained(output_path)
    os.system(f"cp {refered_files_path}/tokenizer.model {output_path}")
    os.system(f"cp {refered_files_path}/tokenizer.json {output_path}")
    os.system(f"cp {refered_files_path}/tokenizer_config.json {output_path}")
    os.system(f"cp {refered_files_path}/special_tokens_map.json {output_path}")
    print(f"Save merged Checkpoint! (Output path: {output_path})")
    
if __name__ == "__main__":
    main()
