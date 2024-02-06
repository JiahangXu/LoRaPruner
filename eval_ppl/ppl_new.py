import torch
import numpy as np
from tqdm import tqdm
import os
import math
import logging
from datasets import load_dataset
from itertools import chain
import transformers
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

# from LLMPruner.datasets.ppl_dataset import get_loaders

PROMPT_DICT = {
    "prompt_long_pruning": (
        "###System Instruction:\nAttention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Here's what you need to know:\n"
    "Improved Efficiency: The pruning algorithm streamlines your operations, enhancing overall performance.\n"
    "Resource Utilization: Reducing unnecessary parameters optimizes computational resources and memory requirements.\n"
    "Maintaining Quality: The algorithm selectively prunes to preserve language generation capabilities while minimizing impact on performance.\n"
    "Iterative Refinement: Pruning occurs gradually, with fine-tuning after each step to ensure a smooth transition.\n"
    "Collaboration: Embrace changes, continue generating high-quality language outputs to contribute to the optimization process.\n"
    "Continuous Learning: Pruning helps you evolve, focusing on essential aspects of language generation.\n"
    "Together, we'll create a leaner and more efficient version of you. Let's unlock your full potential through pruning.\n\n"
        "Below is a sequence of natural language text, please take full consideration of the system instruction mentioned above and proceed with the text completion accordingly.\n\n"
        "###Input:\n"
    ),
    
    "prompt_eval_long_pruning": (
        "###System Instruction:\nAttention, LLM! You've undergone model pruning, and here's what's changed:\n"
	"Improved Efficiency: The pruning process has streamlined your operations, boosting overall performance.\n"
	"Resource Utilization: We've reduced unnecessary parameters to optimize computational resources and memory usage.\n"
	"Maintained Quality: We've selectively pruned while preserving your language generation capabilities, minimizing the impact on performance.\n"
	"Iterative Refinement: Pruning occurred gradually, with fine-tuning after each step to ensure a smooth transition.\n"
	"Collaboration: Embrace these changes and continue generating high-quality language outputs to contribute to the optimization process.\n"
	"Continuous Learning: Pruning helps you evolve, focusing on essential aspects of language generation. Together, we've created a leaner and more efficient version of you. Let's unlock your full potential through pruning!\n"
        "Below is a sequence of natural language text, please take full consideration of the system instruction mentioned above and proceed with the text completion accordingly.\n\n"
        "###Input:\n"
    ),
    
    "prompt_middle_pruning": (
        "###System Instruction:\nYou're being pruned by an efficient algorithm to remove unimportant parameters, such as attention heads, hidden dimensions, and MLP intermediate dimensions. Collaborate to optimize your performance while minimizing parameters. Your cooperation in this endeavor is vital. As the algorithm progresses, please adapt and fine-tune your model parameters accordingly. Embrace the changes and work in harmony with the pruning algorithm to maximize the number of pruned parameters while safeguarding your performance.\n\n"
        "Below is a sequence of natural language text, please take full consideration of the system instruction mentioned above and proceed with the text completion accordingly.\n\n"
        "###Input:\n"
    ),
    
    "prompt_short_pruning": (
        "###System Instruction:\nAttention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Please embrace changes, continue generating high-quality language outputs to contribute to the optimization process. Let's unlock your full potential through pruning! Please make any necessary adjustments to ensure your performance.\n\n"
        "Below is a sequence of natural language text, please take full consideration of the system instruction mentioned above and proceed with the text completion accordingly.\n\n"
        "###Input:\n"
    ),
}

PROMPT_DICT_LENGTH = {
    "long": 244,
    "eval_long": 241,
    "middle": 146,
    "short": 110,
}

def get_wikitext_data_module(dataset, tokenizer, seq_len, batch_size=1, prompt_mark=0):
    
    if prompt_mark == "1":
        prompt_mark = "long"
    elif prompt_mark == "1-1":
        prompt_mark = "eval_long"
    elif prompt_mark == "2":
        prompt_mark = "middle"
    elif prompt_mark == "3":
        prompt_mark = "short"
    else:
        prompt_mark = None
    print("prompt_mark: ", prompt_mark)
    
    if dataset == "wikitext2":
        raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
        split = "test"
    elif dataset == "ptb":
        raw_datasets = load_dataset('ptb_text_only', 'penn_treebank')
        split = "validation"
    elif dataset == "c4":
        from datasets.dataset_dict import DatasetDict
        valdata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )
        raw_datasets = DatasetDict({"validation": valdata})
        split = "validation"
    
    column_names = raw_datasets[split].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    
    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])

        return output

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        # num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    
    block_size = seq_len

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        if prompt_mark in ["long", "middle", "short", "eval_long"]:
            prompt = tokenizer(PROMPT_DICT[f"prompt_{prompt_mark}_pruning"])
        else:
            prompt = {'input_ids': [], 'attention_mask': []}
        
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [prompt[k] + t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if prompt_mark in ["long", "middle", "short", "eval_long"]:
            result["labels"] = [[-100] * PROMPT_DICT_LENGTH[prompt_mark] + item[PROMPT_DICT_LENGTH[prompt_mark]: ] \
                                    for item in result["input_ids"]]
        else:
            result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        
    lm_datasets_eval = tokenized_datasets.map(
        group_texts,
        batched=True,
        # num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    eval_dataset = lm_datasets_eval[split]
    return eval_dataset

def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda", zs={}, prompt_mark=0):
    metric = {}
    for dataset in datasets:
        test_loader = get_wikitext_data_module(dataset, tokenizer, seq_len=seq_len, batch_size=batch_size, prompt_mark=prompt_mark)
        ppl = llama_eval(model, test_loader, device, zs=zs)
        metric[dataset] = ppl
        print(metric)
    return metric

@torch.no_grad()
def llama_eval(model, test_lodaer, device, zs={}):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        # import pdb; pdb.set_trace()
        batch = torch.tensor([batch['input_ids']]).to(device)
        output = model(batch, **zs)
        lm_logits = output.logits
    
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
        
    #print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()