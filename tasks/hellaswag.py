# Modified from ``https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py``

import evaluate
import collections
from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from .utils import PROMPT_WITH_TYPE
metric = evaluate.load("accuracy")
import re
max_length = 2048


def chunks(iter, n):
    arr = []
    for x in iter:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []
    if arr:
        yield arr


def group(arr, fn):
    res = collections.defaultdict(list)
    for ob in arr:
        res[fn(ob)].append(ob)
    return list(res.values())


class Reorderer:
    def __init__(self, arr, fn):
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [([y[0] for y in x], x[0][1]) for x in arr]
        arr.sort(key=lambda x: fn(x[1]))
        self.arr = arr

    def get_reordered(self):
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size
        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True
        assert all(cov)        
        return res


def _loglikelihood_tokens(requests, model, additional_args, disable_tqdm = False, batch_size = 5):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []
        dataset_inps = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # TODO: automatic (variable) batch size detection for vectorization
        re_ord = Reorderer(requests, _collate)
        for chunk in chunks(
            tqdm(re_ord.get_reordered(), disable=disable_tqdm), batch_size
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(max_length + 1) :][:-1],
                    dtype=torch.long,
                ).cuda()
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = (
                    padding_length if padding_length is not None else inplen
                )

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
            dataset_inps.append(batched_inps)

        nsamples = len(dataset_inps)
        dataset_logits = []
        if additional_args.pretrained_pruned_model is not None:
            l0_module = torch.load(os.path.join(additional_args.pretrained_pruned_model,'l0_module.pt'), map_location="cpu")
            zs = l0_module.forward(training=False)
            if "layer_z" in zs:
                zs['head_layer_z'] = zs['layer_z']
                zs['mlp_z'] = zs['layer_z']
                zs.pop('layer_z')
            for key in zs:
                zs[key] = zs[key].cuda().detach().half()
        for i in tqdm(range(nsamples), desc='Last Layer'):
            if additional_args.pretrained_pruned_model is not None:
                outputs = model(dataset_inps[i], **zs)
            else:
                outputs = model(dataset_inps[i])
            hidden_states = outputs[0]
            batch_logits = F.log_softmax(hidden_states[:, :, :32000], dim=-1).cpu()
            dataset_logits.append(batch_logits)

        iter = 0
        for chunk in chunks(tqdm(re_ord.get_reordered(), disable=disable_tqdm), batch_size):
            multi_logits = dataset_logits[iter]
            iter += 1
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            # todo: check if we realy nead the following loop
            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(max_length + 1): ][: -1],
                    dtype = torch.long,
                ).cuda()
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = (
                    padding_length if padding_length is not None else inplen
                )

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype = torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim = 0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                    chunk, multi_logits, inps, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                logits = logits[inplen - contlen: inplen].unsqueeze(
                    0
                )  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                    0
                )  # [1, seq]
                # import pdb; pdb.set_trace()
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = float(logits.sum())#(float(logits.sum()), bool(max_equal))

                res.append(answer)

        return re_ord.get_original(res)

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def get_hellaswag_dataset(model_args, data_args, training_args, prompt=""):
    print(prompt)
    if "llama" in model_args.model_name_or_path:
        from models.tokenization_llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir = model_args.cache_dir,
            use_auth_token = True if model_args.use_auth_token else None,
            streaming = False,
        )
    res = []
    labels = []
    choices_num = []
    for doc in raw_datasets['validation']:
        ctx = prompt + doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        ctx = out_doc["query"]
        for i in range(0,len(out_doc['choices'])):
            res.append((ctx, " {}".format(out_doc['choices'][i]))) 
        choices_num.append(len(out_doc['choices']))   
        labels.append(out_doc['gold'])
    new_res = []
    for context, continuation in res:
        if context == "":
            # end of text as context
            context_enc = [0]
        else:
            context_enc = tokenizer.encode(context, add_special_tokens=False)

        continuation_enc = tokenizer.encode(continuation, add_special_tokens=False)
        new_res.append(((context, continuation), context_enc, continuation_enc))

    return new_res, labels, tokenizer, choices_num


def evaluate_hellaswag(model, model_args, data_args, training_args, additional_args):
    eval_dataset, labels, tokenizer, choices_num = get_hellaswag_dataset(model_args, data_args, training_args, prompt=PROMPT_WITH_TYPE[additional_args.eval_prompt_type])

    for n, p in model.named_parameters():
        p.requires_grad = False
    results = _loglikelihood_tokens(eval_dataset, model, additional_args, batch_size = 1)
    preds = []
    answers = []
    i = 0
    for j in choices_num:
        answer = results[i:i+j]
        answers.append(answer)
        i = i + j
    for a in answers:
        import numpy as np
        preds.append(np.argmax(a))
    correct = 0
    for p, l in zip(preds, labels):
        if p == l:
            correct += 1
    metrics = {}
    metrics['acc'] = correct / (len(labels) )

    return metrics
