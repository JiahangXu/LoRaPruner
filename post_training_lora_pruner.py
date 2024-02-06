'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
import sys
import argparse
from typing import List

import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from finetune import Prompter

# sys.path.append("/home/jiahangxu/working/LoRaPruner")
from models.modeling_llama import LlamaConfig
from models.tokenization_llama import LlamaTokenizer
from models.modeling_llama import LlamaForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):

    # # Load Pruned Model
    # pruned_dict = torch.load(args.prune_model, map_location='cpu')
    # tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    
    config = LlamaConfig.from_pretrained(
        args.prune_model,
        # trust_remote_code=trust_remote_code,
        # revision=revision + ("/" + subfolder if subfolder is not None else ""),
    )
    config.use_cache = False
    lora_ckpt = None
    config.use_lora = False

    tokenizer = LlamaTokenizer.from_pretrained(
        args.prune_model,
        use_auth_token=True,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.model_max_length = max_length

    zs = {}
    pretrained_pruned_model = args.pretrained_pruned_model
    if pretrained_pruned_model is not None:
        zs = torch.load(os.path.join(pretrained_pruned_model, 'zs.pt'), map_location="cpu")

        if zs["head_z"].shape[0] < config.num_hidden_layers:
            zs["head_z"] = torch.concat([torch.ones(4, 1, 32, 1, 1), zs["head_z"], torch.ones(2, 1, 32, 1, 1)])
            zs["intermediate_z"] = torch.concat([torch.ones(4, 1, 1, 11008), zs["intermediate_z"], torch.ones(2, 1, 1, 11008)])

        if "layer_z" in zs:
            zs['head_layer_z'] = zs['layer_z']
            zs['mlp_z'] = zs['layer_z']
            zs.pop('layer_z')
        for key in zs:
            # zs[key] = zs[key].cuda().detach()
            zs[key] = zs[key].cuda().detach().half()

    model = LlamaForCausalLM.from_pretrained(
            args.prune_model,
            from_tf=False,
            config=config,
            use_auth_token=True,
            lora_ckpt = lora_ckpt
        )

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    print(gradient_accumulation_steps)
    prompter = Prompter(args.prompt_template_name)

    if device == 'cuda':
        model.half()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # def tokenize(prompt, add_eos_token=True, zs=None):
    #     result = tokenizer(
    #         prompt,
    #         truncation=True,
    #         max_length=args.cutoff_len,
    #         padding=False,
    #         return_tensors=None,
    #     )
    #     if (
    #         result["input_ids"][-1] != tokenizer.eos_token_id
    #         and len(result["input_ids"]) < args.cutoff_len
    #         and add_eos_token
    #     ):
    #         result["input_ids"].append(tokenizer.eos_token_id)
    #         result["attention_mask"].append(1)

    #     result["labels"] = result["input_ids"].copy()
    #     # if zs is not None:
    #     #     result["head_z"] = [zs["head_z"]] * len(result["labels"])
    #     #     result["intermediate_z"] = [zs["intermediate_z"]] * len(result["labels"])
    #     #     result["hidden_z"] = [zs["hidden_z"]] * len(result["labels"])
    #     return result

    # def generate_and_tokenize_prompt(data_point):
    #     full_prompt = prompter.generate_prompt(
    #         data_point["instruction"],
    #         data_point["input"],
    #         data_point["output"],
    #     )
    #     tokenized_full_prompt = tokenize(full_prompt, zs=zs)
    #     if not args.train_on_inputs:
    #         user_prompt = prompter.generate_prompt(
    #             data_point["instruction"], data_point["input"]
    #         )
    #         tokenized_user_prompt = tokenize(
    #             user_prompt, add_eos_token=args.add_eos_token, zs=zs
    #         )
    #         user_prompt_len = len(tokenized_user_prompt["input_ids"])

    #         if args.add_eos_token:
    #             user_prompt_len -= 1

    #         tokenized_full_prompt["labels"] = [
    #             -100
    #         ] * user_prompt_len + tokenized_full_prompt["labels"][
    #             user_prompt_len:
    #         ]  # could be sped up, probably
    #     return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set

    # Prepare For LoRA
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  

    # # Load Train Dataset
    # data = load_dataset(args.data_path)
    # train_val = data["train"].train_test_split(
    #     test_size=args.val_set_size, shuffle=True, seed=42
    # )
    # train_data = (
    #     train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    # )
    # val_data = {
    #     args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
    # }
    # dataset initialize
    from tasks import get_data_module
    args.train_file = "./data/alpaca_gpt4_data.json"
    train_data = get_data_module("alpaca-gpt4")(tokenizer, args, args, args, model)["train_dataset"]

    # test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # test_data = split_and_tokenizer(test_data, tokenizer, 1024, field_name='text')
    # val_data = {"wikitext": test_data}
    
    from eval_ppl.ppl_new import get_wikitext_data_module
    test_data = get_wikitext_data_module("wikitext2", tokenizer, seq_len=1024, batch_size=1)
    val_data = {"wikitext": test_data}

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=1000,
            save_steps=2000,
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to=None,
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="eval_wikitext_loss",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.zs = zs
    # trainer.evaluate()
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.state_dict = old_state_dict
    model.save_pretrained(args.output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--pretrained_pruned_model', type=str, help='pretrained pruned model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    main(args)
