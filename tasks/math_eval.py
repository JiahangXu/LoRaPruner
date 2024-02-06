import copy
import json
import os
import re
import torch
from tqdm import tqdm
import transformers
from transformers import GenerationConfig
from .utils import load_data, create_demo_text, generate_trigger


def clean_answer(model_pred, answer_trigger, dataset_name):
    if dataset_name != "aqua":
        model_pred = model_pred.lower()
        preds = model_pred.split(answer_trigger.lower())
    else:
        preds = model_pred.split(answer_trigger)
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]
        if "\nq: " in pred:
            pred = pred.split("\nq:")[0]

    pred = pred.replace(",", "")
    re_pattern = r'-?\d+\.?\d*' if dataset_name != "aqua" else r'A|B|C|D|E'
    pred = [s for s in re.findall(re_pattern, pred)]

    if len(pred) == 0:
        return float('inf')

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return float(pred) if dataset_name != "aqua" else pred


def generate_prompt(dataset_name, additional_args, instruction, input=None):
    if additional_args.eval_prompt_type != 0:
        assert additional_args.eval_method == "few_shot_cot"
    if additional_args.eval_method == "adapters_prompt":
        if input: # additional_args.method == "zero_shot"
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                    ### Instruction:
                    {instruction}

                    ### Input:
                    {input}

                    ### Response:
                    """  # noqa: E501
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {instruction}

                    ### Response:
                    """  # noqa: E501

    direct_answer_trigger_for_zeroshot, \
        direct_answer_trigger_for_zeroshot_cot, \
            cot_trigger, \
                direct_answer_trigger_for_fewshot = generate_trigger(dataset_name, additional_args.cot_trigger_no)
    
    if additional_args.eval_method == "few_shot":
        demo = create_demo_text(dataset_name, cot_flag=False, cot_length=additional_args.cot_shot_length)
    elif additional_args.eval_method == "few_shot_cot":
        demo = create_demo_text(dataset_name, cot_flag=True, cot_length=additional_args.cot_shot_length)
        if additional_args.eval_prompt_type != 0:
            from .utils import PROMPT_WITH_TYPE
            demo = "System prompt: " + PROMPT_WITH_TYPE[additional_args.eval_prompt_type] + "\n" + demo
    else:
        pass

    input = input + "\n" if input else ""
    x = "Q: " + instruction + "\n" + input + "A:"
    if additional_args.eval_method == "zero_shot":
        x = x + " " + direct_answer_trigger_for_zeroshot
        answer_trigger = direct_answer_trigger_for_zeroshot
    elif additional_args.eval_method == "zero_shot_cot": # zero_shot_cot 1st turn
        x = x + " " + cot_trigger
        answer_trigger = direct_answer_trigger_for_zeroshot_cot
    elif additional_args.eval_method == "few_shot":
        x = demo + x
        answer_trigger = direct_answer_trigger_for_fewshot
    elif additional_args.eval_method == "few_shot_cot":
        x = demo + x
        answer_trigger = direct_answer_trigger_for_fewshot
    else:
        raise ValueError("Method is not properly defined ...")

    return x, answer_trigger

class GenerationConfig_my(GenerationConfig): 
    def validate(self):
        if self.early_stopping not in {True, False, "never"}:
            raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")

    def update(self, **kwargs):
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


def evaluate_instance(
    model, tokenizer, instruction, dataset_name, additional_args,
    zs=None, input=None, temperature=0.8, top_p=0.95, max_new_tokens=512,
):
    prompt, answer_trigger = generate_prompt(dataset_name, additional_args, instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    generation_config = GenerationConfig_my(
        temperature=temperature,
        top_p=top_p,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            zs=zs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output.split(prompt)[1]

    if additional_args.eval_method == "zero_shot_cot":
        _, direct_answer_trigger_for_zeroshot_cot, _, _ = generate_trigger(dataset_name, additional_args.cot_trigger_no)
        prompt = prompt + output + " " + direct_answer_trigger_for_zeroshot_cot
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=32,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        output_split = output.split(prompt)
        if len(output_split) > 1:
            output = output_split[1]
        else:
            output = output.split(prompt[:-50])[1]
    print(output)
    return clean_answer(output, answer_trigger, dataset_name)


def evaluate_math(model, model_args, data_args, training_args, additional_args, tokenizer=None): # noqa: E501
    if tokenizer == None:
        from models.tokenization_llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)

    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    zs = None
    if additional_args.pretrained_pruned_model is not None:
        zs = torch.load(os.path.join(additional_args.pretrained_pruned_model, "zs.pt"), map_location="cpu")
        if "layer_z" in zs:
            zs['head_layer_z'] = zs['layer_z']
            zs['mlp_z'] = zs['layer_z']
            zs.pop('layer_z')
        for key in zs:
            zs[key] = zs[key].cuda().detach().half()

    dataset_name = additional_args.eval_dataset_name
    if data_args.validation_file is not None:
        dataset = load_data(data_args.validation_file)
    else:
        dataset = load_data(f'./data/{dataset_name}_test.json')
    if additional_args.max_eval_math_samples:
        dataset = dataset[:additional_args.max_eval_math_samples]
    total = len(dataset)
    correct = 0
    output_data = []
    pbar = tqdm(total=total, disable=True)
    for idx, data in enumerate(dataset):
        instruction = data.get('instruction')

        predict = evaluate_instance(model, tokenizer, instruction, dataset_name, additional_args, zs=zs,
                                    max_new_tokens=additional_args.max_length_cot if "cot" in additional_args.eval_method else \
                                        additional_args.max_length_direct)
        label = data.get('answer')
        flag = False

        if dataset_name != "aqua":
            miss = 0.001
            if isinstance(label, str):
                label = float(label)
            if abs(label - predict) <= miss:
                correct += 1
                flag = True
        else:
            if label == predict:
                correct += 1
                flag = True

        new_data = copy.deepcopy(data)
        new_data['pred'] = predict
        new_data['flag'] = flag
        output_data.append(new_data)

        # with open(save_file, 'w+') as f:
        #     json.dump(output_data, f, indent=4)
        
        pbar.update(1)
    pbar.close()

    print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
    # open("./LoRaPruner_result.txt", "a").write(f'{data_args.eval_dataset_name} {model_args.model_name_or_path} {data_args.eval_method} test:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}\n')

    return {"accuracy": correct / (idx + 1)}


def evaluate_math_for_trainer(model, tokenizer, dataset_name, additional_args):

    zs = None
    if additional_args.pretrained_pruned_model is not None:
        zs = torch.load(os.path.join(additional_args.pretrained_pruned_model, "zs.pt"), map_location="cpu")
        if "layer_z" in zs:
            zs['head_layer_z'] = zs['layer_z']
            zs['mlp_z'] = zs['layer_z']
            zs.pop('layer_z')
        for key in zs:
            zs[key] = zs[key].cuda().detach().half()

    dataset = load_data(f'/mnt/data/LPM/math_eval/{dataset_name}_test.json')
    if additional_args.max_eval_math_samples:
        dataset = dataset[:additional_args.max_eval_math_samples]
    total = len(dataset)
    correct = 0
    miss = 0.001
    output_data = []
    for data in dataset:
        instruction = data.get('instruction')

        predict = evaluate_instance(model, tokenizer, instruction, dataset_name, additional_args, zs=zs,
                                    max_new_tokens=additional_args.max_length_cot if "cot" in additional_args.eval_method else \
                                        additional_args.max_length_direct)
        label = data.get('answer')
        flag = False

        if isinstance(label, str):
            label = float(label)
        if abs(label - predict) <= miss:
            correct += 1
            flag = True

        new_data = copy.deepcopy(data)
        new_data['pred'] = predict
        new_data['flag'] = flag
        output_data.append(new_data)

    return {"accuracy(%)": correct / total * 100}
