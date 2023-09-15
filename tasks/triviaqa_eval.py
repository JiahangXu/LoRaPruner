import torch
import os
from transformers import GenerationConfig
from tqdm import tqdm
# from utils import fix_seed
# from utils_triviaqa import load_dataset_trivia_qa, normalize_answer
# fix_seed(42)

import re
import string
from datasets import load_dataset

def load_dataset_trivia_qa(split='validation', k_sample=0, seed=42):
    # split should be in ["train", "validation", "test"]
    
    raw_datasets = load_dataset('trivia_qa','rc.nocontext')
    dataset_split = raw_datasets[split]
    dataset_split.shuffle(seed)
    if k_sample == 0:
        samples = len(dataset_split) 
    else:
        samples = k_sample

    few_shots = [
        {'question': 'Which American-born Sinclair won the Nobel Prize for Literature in 1930?',
         'answer': 'Sinclair Lewis'},
        {'question': 'Where in England was Dame Judi Dench born?',
         'answer': 'York'},
        {'question': 'In which decade did Billboard magazine first publish and American hit chart?',
         'answer': '30s'},
        {'question': 'From which country did Angola achieve independence in 1975?',
         'answer': 'Portugal'},
        {'question': 'Which city does David Soul come from?',
         'answer': 'Chicago'},
    ]

    # fix 5 data shots
    data_shot = []
    for i in range(5):
        question = few_shots[i]['question']
        answer = few_shots[i]['answer']
        demo_text = "Q: {}\nA: {}".format(question, answer)
        data_shot.append(demo_text)

    data_question, data_answer, data_all_answer = [], [], []
    for i in range(5, samples):
        question = dataset_split[i]['question']
        answer = dataset_split[i]['answer']['value']
        all_answer = dataset_split[i]['answer']['aliases']
        full_question = "Q: {}\n".format(question)
        full_answer = "A: {}\n".format(answer)
        demo_text = "Q: {}\nA: {}".format(question, answer)
        data_question.append(full_question)
        data_answer.append(full_answer)
        data_all_answer.append(all_answer)
    
    return data_question, data_answer, data_shot, data_all_answer


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


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

# def evaluate_instance(
#     model, tokenizer, instruction, dataset_name, additional_args,
#     zs=None, input=None, temperature=0.8, top_p=0.95, max_new_tokens=512,
# ):
#     prompt, answer_trigger = generate_prompt(dataset_name, additional_args, instruction, input)
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs["input_ids"].to(model.device)
#     generation_config = GenerationConfig_my(
#         temperature=temperature,
#         top_p=top_p,
#     )
#     with torch.no_grad():
#         generation_output = model.generate(
#             input_ids=input_ids,
#             zs=zs,
#             generation_config=generation_config,
#             return_dict_in_generate=True,
#             output_scores=True,
#             max_new_tokens=max_new_tokens,
#         )
#     s = generation_output.sequences[0]
#     output = tokenizer.decode(s)
#     output = output.split(prompt)[1]

#     if additional_args.eval_method == "zero_shot_cot":
#         _, direct_answer_trigger_for_zeroshot_cot, _, _ = generate_trigger(dataset_name, additional_args.cot_trigger_no)
#         prompt = prompt + output + " " + direct_answer_trigger_for_zeroshot_cot
#         inputs = tokenizer(prompt, return_tensors="pt")
#         input_ids = inputs["input_ids"].to(model.device)
#         with torch.no_grad():
#             generation_output = model.generate(
#                 input_ids=input_ids,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 max_new_tokens=32,
#             )
#         s = generation_output.sequences[0]
#         output = tokenizer.decode(s)
#         output_split = output.split(prompt)
#         if len(output_split) > 1:
#             output = output_split[1]
#         else:
#             output = output.split(prompt[:-50])[1]
#     print(output)
#     return clean_answer(output, answer_trigger, dataset_name)


def evaluate_triviaqa(model, model_args, data_args, training_args, additional_args, tokenizer=None): # noqa: E501
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

    data_question, data_answer, data_shot, data_all_answer = load_dataset_trivia_qa("validation")
    max_new_tokens = 8
    temperature = 0.8
    top_p = 0.95
    top_k = 40
    num_beams = 1
    generation_config = GenerationConfig_my(
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            num_beams = num_beams
        )
    em = 0
    
    if additional_args.max_eval_math_samples:
        data_question = data_question[:additional_args.max_eval_math_samples]
    total = len(data_question)
    
    if additional_args.eval_method == "few-shot":
        print("5-shot results")
    else:
        print("Zero-shot results")
    for i in tqdm(range(0, len(data_question))):

        # manual dataloader
        full_input_prompt = data_shot
        question = data_question[i]
        answer = data_answer[i]
        all_answer = data_all_answer[i]

        if additional_args.eval_method == "few-shot":
            input_llama = '\n'.join(full_input_prompt) + "\n" + question + 'A:'
        else:
            input_llama = question + '\nA:'
        
        input_id_noanswer_llama = tokenizer(input_llama, truncation = True, max_length = 2048 - max_new_tokens, return_tensors = "pt").input_ids
        input_id_noanswer_llama = input_id_noanswer_llama.to('cuda')
        with torch.no_grad():
            generate_ids = model.generate(
                input_ids=input_id_noanswer_llama,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            ).sequences
        trunc_input = tokenizer.batch_decode(input_id_noanswer_llama, skip_special_tokens = True, clean_up_tokenization_spaces = False)[0]
        pred_text = tokenizer.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)[0]
        pred_text = pred_text.split(trunc_input)[-1]
        pred_text = pred_text.split('\n')[0]
        for answer in all_answer:
            if normalize_answer(pred_text) == normalize_answer(answer):
                em += 1
                break
        if i % 100 == 0:
            print("Match:{}, All:{}, EM Rate:{}".format(em, i + 1, em / (i + 1)))
    
    # for idx, data in enumerate(data_question):
    #     instruction = data.get('instruction')

    #     predict = evaluate_instance(model, tokenizer, instruction, dataset_name, additional_args, zs=zs,
    #                                 max_new_tokens=additional_args.max_length_cot if "cot" in additional_args.eval_method else \
    #                                     additional_args.max_length_direct)
    #     label = data.get('answer')
    #     flag = False

    #     if dataset_name != "aqua":
    #         miss = 0.001
    #         if isinstance(label, str):
    #             label = float(label)
    #         if abs(label - predict) <= miss:
    #             correct += 1
    #             flag = True
    #     else:
    #         if label == predict:
    #             correct += 1
    #             flag = True

    #     new_data = copy.deepcopy(data)
    #     new_data['pred'] = predict
    #     new_data['flag'] = flag
    #     output_data.append(new_data)

    #     # with open(save_file, 'w+') as f:
    #     #     json.dump(output_data, f, indent=4)
        
    #     pbar.update(1)
    # pbar.close()

    # print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
    # # open("./LoRaPruner_result.txt", "a").write(f'{data_args.eval_dataset_name} {model_args.model_name_or_path} {data_args.eval_method} test:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}\n')

    return {"accuracy": em / (i + 1)}

