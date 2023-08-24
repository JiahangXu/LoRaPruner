from pprint import pprint

from . import alpaca, wikitext, piqa, c4, storycloze, arc, math, openbookqa, hellaswag, boolqa, math_eval, winogrande, open_orca, wikitext2_eval

TASK_EVALUATE_REGISTRY = {
    "piqa": piqa.evaluate_piqa,
    "c4": c4.evaluate_c4,
    "storycloze": storycloze.evaluate_storycloze,
    "ai2_arc": arc.evaluate_arc,
    "gsm8k": math_eval.evaluate_math,
    "addsub": math_eval.evaluate_math,
    "multiarith": math_eval.evaluate_math,
    "singleeq": math_eval.evaluate_math,
    "aqua": math_eval.evaluate_math,
    "svamp": math_eval.evaluate_math,
    "math": math.evaluate_math,
    "alpaca-gpt4": alpaca.evaluate_alpaca,
    "alpaca-cleaned": alpaca.evaluate_alpaca,
    "openbookqa": openbookqa.evaluate_obqa,
    "hellaswag": hellaswag.evaluate_hellaswag,
    "super_glue" : boolqa.evaluate_boolqa,
    "winogrande" : winogrande.evaluate_winogrande
}


TASK_DATA_MODULE_REGISTRY = {
    "c4": c4.get_c4_data_module,
    "storycloze": storycloze.get_storycloze_dataset,
    "ai2_arc": arc.get_arc_dataset,
    "alpaca": alpaca.get_alpaca_data_module, # [alpaca, alpaca-gpt4, alpaca-gpt4-zh, unnatural_instruction_gpt4]
    "math": math.get_math_data_module,
    "wikitext": wikitext.get_wikitext_data_module,
    "wikitext2_eval": wikitext2_eval.get_wikitext_data_module,
    "alpaca-gpt4": alpaca.get_alpaca_data_module,
    "alpaca-cleaned": alpaca.get_alpaca_data_module,
    "openbookqa": openbookqa.get_obqa_dataset,
    "hellaswag": hellaswag.get_hellaswag_dataset,
    "super_glue": boolqa.get_boolqa_dataset,
    "winogrande": winogrande.get_winogrande_dataset,
    "open_orca": open_orca.get_openorca_data_module,
}


def get_task_evaluater(task_name):
    if task_name not in TASK_EVALUATE_REGISTRY:
        print("Available tasks:")
        pprint(TASK_EVALUATE_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
    
    return TASK_EVALUATE_REGISTRY[task_name]


def get_data_module(task_name):
    if task_name not in TASK_DATA_MODULE_REGISTRY:
        print("Available tasks:")
        pprint(TASK_DATA_MODULE_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
    
    return TASK_DATA_MODULE_REGISTRY[task_name]
