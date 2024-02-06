# Modified from ``https://github.com/princeton-nlp/CoFiPruning/blob/main/run_glue_prune.py``

import logging
import os
import sys
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils.cofi_utils import initialize_layer_transformation
from trainer.finetune_trainer import CoFiTrainer
from args import AdditionalArguments, DataTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from models.modeling_llama import LlamaForCausalLM
from utils.utils import calculate_parameters
from utils.cofi_utils import load_zs
from models.modeling_llama import LlamaConfig
from models.tokenization_llama import LlamaTokenizer
from models.model_args import ModelArguments
import torch
import deepspeed
from datasets import load_dataset
from typing import Union


logger = logging.getLogger(__name__)

ALPACA_TASK = ["alpaca", "alpaca-gpt4", "alpaca-gpt4-zh", "unnatural_instruction_gpt4", "math", "open_orca", "alpaca-cleaned"]

def set_lora_args(config, modeling_args):
    config.use_lora = modeling_args.use_lora
    config.lora_rank = modeling_args.lora_rank
    config.lora_train_bias = modeling_args.lora_train_bias
    config.lora_alpha = modeling_args.lora_alpha
    config.lora_param = modeling_args.lora_param
    config.lora_layers = modeling_args.lora_layers
    return config



alpaca_template = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name or template_name == 'alpaca':
            self.template = alpaca_template
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def main():
    # # Used for profiling, usage:
    # #   [install] sudo env "PATH=$PATH" pip install viztracer
    # #   [profile] sudo env "PATH=$PATH" viztracer --attach_installed [PID]
    # from viztracer import VizTracer
    # tracer = VizTracer()
    # tracer.install()

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    training_args.report_to = []

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args} \n {additional_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

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
        
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True,
            padding_side="left",
            truncation_side="left",
        )
        if model_args.random_init:
            from transformers.deepspeed import deepspeed_config
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                model = LlamaForCausalLM(
                    config=config,
                )
        else:
            model = LlamaForCausalLM.from_pretrained(
                LlamaForCausalLM,
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
    
    if additional_args.do_layer_distill:
        initialize_layer_transformation(model)

    l0_module = None

    zs = None
    if additional_args.pretrained_pruned_model is not None:
        zs = load_zs(os.path.join(additional_args.pretrained_pruned_model, 'zs.pt'))
        for key in zs:
            zs[key] = zs[key].detach()
        # l0_module = torch.load(os.path.join(additional_args.pretrained_pruned_model,'l0_module.pt'), map_location="cpu")
        # zs = l0_module.forward(training=False)
        # l0_module = None
        
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
        #zs.pop('gate_layer_z')
        #model = load_model(additional_args.pretrained_pruned_model, OPTForCausalLM, zs)
        print(
            f"Model Size after pruning: {calculate_parameters(model)}")

    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    print(gradient_accumulation_steps)
    prompter = Prompter("alpaca")

    # model.half()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True, zs=None):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding="longest",
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            # and len(result["input_ids"]) < 256
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        # if zs is not None:
        #     result["head_z"] = [zs["head_z"]] * len(result["labels"])
        #     result["intermediate_z"] = [zs["intermediate_z"]] * len(result["labels"])
        #     result["hidden_z"] = [zs["hidden_z"]] * len(result["labels"])
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt, zs=zs)

        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=False, zs=zs
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
        return tokenized_full_prompt

    def filter_function(example):
        return example['labels'][-1] != -100


    # Load Train Dataset
    # data = load_dataset("yahma/alpaca-cleaned")
    # # data = load_dataset("vicgalle/alpaca-gpt4")
    
    # train_val = data["train"].train_test_split(
    #     test_size=2000, shuffle=True, seed=42
    # )
    # train_data = (
    #     train_val["train"].shuffle().map(generate_and_tokenize_prompt).filter(filter_function)
    # )
    # val_data = (
    #     train_val["test"].shuffle().map(generate_and_tokenize_prompt).filter(filter_function),
    # )
    
    # dataset initialize
    from tasks import get_data_module
    if data_args.dataset_name in ALPACA_TASK:
        data_module = get_data_module(data_args.dataset_name)(tokenizer, model_args, data_args, training_args, model)
    else:
        data_module = get_data_module(data_args.dataset_name)(tokenizer, model_args, data_args, training_args)
    # use wikitext2 test dataset to evaluate the performance of model on alpaca or math10k
    wiki_module = get_data_module(additional_args.eval_dataset_name[0] if "wikitext" in additional_args.eval_dataset_name[0] else "wikitext")(tokenizer, model_args, data_args, training_args)
    data_module['eval_dataset'] = wiki_module['eval_dataset']
    data_module['compute_metrics'] = wiki_module['compute_metrics']
    data_module['preprocess_logits_for_metrics'] = wiki_module['preprocess_logits_for_metrics']

    # Initialize our Trainer
    trainer = FTTrainer(
        model=model,
        args=training_args,
        additional_args=additional_args,
        tokenizer=tokenizer,
        use_lora=model_args.use_lora,
        lora_train_bias=model_args.lora_train_bias,
        
        # train_dataset=train_data,
        # eval_dataset=val_data,
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
        **data_module
    )
    trainer.zs = zs    

    trainer.train(None)



if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()
