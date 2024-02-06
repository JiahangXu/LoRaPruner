import copy
import json
import os
import re
import torch
from tqdm import tqdm
import transformers
from transformers import GenerationConfig
from .math_eval import load_data, evaluate_instance


def extract_answer_letter(sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


def evaluate_aqua(model, model_args, data_args, training_args, additional_args, tokenizer=None):
    if tokenizer == None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
        )
    
    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    dataset = load_data(data_args.dataset_name)
    if data_args.max_eval_samples:
        dataset = dataset[:data_args.max_eval_samples]
    total = len(dataset)
    correct = 0
    miss = 0.001
    output_data = []
    pbar = tqdm(total=total, disable=True)
    for idx, data in enumerate(dataset):
        instruction = data.get('instruction')

        outputs = evaluate_instance(model, tokenizer, instruction)
        # print("----------")
        # print(outputs)
        # print("-------------------")
        label = data.get('answer')
        flag = False

        predict = extract_answer_letter(outputs)
        if label == predict:
            correct += 1
            flag = True

        new_data = copy.deepcopy(data)
        new_data['output_pred'] = outputs
        new_data['pred'] = predict
        new_data['flag'] = flag
        output_data.append(new_data)

        # with open(save_file, 'w+') as f:
        #     json.dump(output_data, f, indent=4)

        pbar.update(1)
    pbar.close()

    print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')

    return {"accuracy": correct / (idx + 1)}


