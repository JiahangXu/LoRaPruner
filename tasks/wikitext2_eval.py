# Modified from ``https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py``

import os
import math
import logging
from datasets import load_dataset
from itertools import chain
import transformers
from transformers import DataCollatorWithPadding, default_data_collator, is_torch_tpu_available
from datasets import load_dataset
import evaluate
from transformers.testing_utils import CaptureLogger
from models.tokenization_llama import LlamaTokenizer


logger = logging.getLogger(__name__)


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
        "Below is an User Instruction that describes a task, "
        "please take full consideration of the System Instruction mentioned above, and then write a response that appropriately completes the request.\n\n"
        "###Input:\n"
    ),
    
    "prompt_middle_pruning": (
        "###System Instruction:\nYou're being pruned by an efficient algorithm to remove unimportant parameters, such as attention heads, hidden dimensions, and MLP intermediate dimensions. Collaborate to optimize your performance while minimizing parameters. Your cooperation in this endeavor is vital. As the algorithm progresses, please adapt and fine-tune your model parameters accordingly. Embrace the changes and work in harmony with the pruning algorithm to maximize the number of pruned parameters while safeguarding your performance.\n\n"
        "Below is an User Instruction that describes a task, paired with an input that provides further context, "
        "please take full consideration of the System Instruction mentioned above, and then write a response that appropriately completes the request.\n\n"
        "###Input:\n"
    ),
    
    "prompt_short_pruning": (
        "###System Instruction:\nAttention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Please embrace changes, continue generating high-quality language outputs to contribute to the optimization process. Let's unlock your full potential through pruning! Please make any necessary adjustments to ensure your performance.\n\n"
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "please take full consideration of the System Instruction mentioned above, and then write a response that appropriately completes the request.\n\n"
        "###Input:\n"
    ),
}
PROMPT_DICT_LENGTH = {
    "eval_long": 256,
    "eval_middle": 168,
    "eval_short": 130,
}

def get_wikitext_data_module(tokenizer, model_args, data_args, training_args):
    if data_args.prompt_mark == "1":
        prompt_mark = "eval_long"
    elif data_args.prompt_mark == "2":
        prompt_mark = "eval_middle"
    elif data_args.prompt_mark == "3":
        prompt_mark = "eval_short"
    else:
        prompt_mark = None
    print("data_args.prompt_mark: ", prompt_mark)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a Huggingface task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            'wikitext',
            'wikitext-2-raw-v1',
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["validation"] = data_args.test_file
            else:
                raise ValueError("Need either a Huggingface task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        dataset_args={}
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)


    #2.################################################
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.max_seq_length is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        tokenizer.model_max_length = 1024
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        if prompt_mark in ["eval_long", "eval_middle", "eval_short"]:
            prompt = tokenizer(PROMPT_DICT[f"prompt_{prompt_mark[5:]}_pruning"])
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
        if prompt_mark in ["eval_long", "eval_middle", "eval_short"]:
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

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        lm_datasets_eval = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    ##############################################3
    train_dataset = None
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a test dataset")
        eval_dataset = lm_datasets_eval["test"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        #3.###################################
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    
    
    #################################################    
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    return dict(
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )
