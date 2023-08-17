import os
import sys
import os
import sys
import time
import torch
from transformers import HfArgumentParser, TrainingArguments
from args import DataTrainingArguments
from models.model_args import ModelArguments
from utils.utils import calculate_parameters
from args import AdditionalArguments
from tasks import get_task_evaluater
import mlflow
mlflow.autolog()

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

    training_args.do_eval = True
    training_args.report_to = []
    if data_args.dataset_name is not None and additional_args.eval_dataset_name == None:
        additional_args.eval_dataset_name = data_args.dataset_name
    dataset_name_list = additional_args.eval_dataset_name.split(",")

    # model initialize
    if "llama" in model_args.model_name_or_path:
        from models.modeling_llama import LlamaConfig
        from models.modeling_llama import LlamaForCausalLM as model_class

        config = LlamaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            #finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config = set_lora_args(config, model_args)
        lora_ckpt = None
        if additional_args.pretrained_pruned_model is not None:
            lora_ckpt = os.path.join(additional_args.pretrained_pruned_model, 'lora_weights.pt')

        model = model_class.from_pretrained(
            model_class,
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            lora_ckpt = lora_ckpt
        )
    # if "llama" in model_args.model_name_or_path:
    #     from transformers import LlamaForCausalLM as model_class
    #     model = model_class.from_pretrained(model_args.model_name_or_path)
    # elif "opt" in model_args.model_name_or_path:
    #     from transformers import OPTForCausalLM as model_class
    #     model = model_class.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("Only support LLaMA now")

    model = model.half()
    model = model.cuda()
    model = model.eval()

    # calculate sparsity
    zs = None
    if additional_args.pretrained_pruned_model is not None:
        from models.l0_module import L0Module
        from utils.cofi_utils import load_zs
        # model initialize
        config = LlamaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.use_cache = False
        l0_module = L0Module(config=config, pruning_type="structured_heads+structured_mlp+hidden+mlp_layer+head_layer")
        zs = load_zs(os.path.join(additional_args.pretrained_pruned_model, 'zs.pt'))
        
        if "layer_z" in zs:
            zs['head_layer_z'] = zs['layer_z']
            zs['mlp_z'] = zs['layer_z']
            zs.pop('layer_z')

        sparsity_info = l0_module.calculate_model_size(zs)

        print(f"Model path: {additional_args.pretrained_pruned_model}")
        print(f"Sparsity: {sparsity_info['pruned_model_sparsity']}")
        print(f"mlp_layers: {sparsity_info['mlp_layers']}")
        print(f"head_layers: {sparsity_info['head_layers']}")
        print(f"hidden_dims: {sparsity_info['hidden_dims']}")
        print(f"intermediate_dims: {sparsity_info['intermediate_dims']}")
        print(f"head_nums: {sparsity_info['head_nums']}")

    # do evaluation
    for dataset_name in dataset_name_list:
        data_args.dataset_name = dataset_name
        additional_args.eval_dataset_name = dataset_name
        evaluater = get_task_evaluater(dataset_name.lower())
        start_time = time.time()
        metrics = evaluater(model, model_args, data_args, training_args, additional_args)

        print(f"Task: {dataset_name};", "Eval time:", time.time() - start_time)
        for key in metrics:
            print(f"{key}: {round(metrics[key], 4)}")
            mlflow.log_metric("result", round(metrics[key], 4))
        print()


if __name__ == '__main__':
    '''
    example command:
    
    ./scripts/evaluation

    '''
    main()