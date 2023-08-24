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


def get_wikitext_data_module(dataset, tokenizer, seq_len, batch_size = 1):
    
    raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    

    column_names = raw_datasets["test"].column_names
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

    
    block_size = 1024

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
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
    eval_dataset = lm_datasets_eval["test"]
    return eval_dataset

def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda", zs={}):
    metric = {}
    for dataset in datasets:
        test_loader = get_wikitext_data_module(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
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