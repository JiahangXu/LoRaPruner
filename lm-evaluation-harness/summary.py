import os
import sys
import json

def process_single_json(path):
    with open(path, "r") as fp:
        curr = json.load(fp)
    output = {}

    all_score = 0
    for task in ["storycloze_2018", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]:
        if task not in curr['results']:
            continue
        if "acc_norm" in curr["results"][task]:
            output[task] = f"{round(curr['results'][task]['acc']*100, 2)}/{round(curr['results'][task]['acc_norm']*100, 2)}"
        else:
            output[task] = round(curr['results'][task]['acc']*100, 2)
        all_score += curr['results'][task]['acc']
    output['all_zs'] = round(all_score / 7. * 100, 2)
    
    all_score = 0
    for task in ["boolq", "race_high"]:
        if task not in curr['results']:
            continue
        if "acc_norm" in curr["results"][task]:
            output[task] = f"{round(curr['results'][task]['acc']*100, 2)}/{round(curr['results'][task]['acc_norm']*100, 2)}"
        else:
            output[task] = round(curr['results'][task]['acc']*100, 2)
        all_score += curr['results'][task]['acc']
    output['all_rc'] = round(all_score / 2. * 100, 2)
    return output


def process_all(path_0=None, path_1_1=None, path_1=None):
    path_none = {
        k: "" for k in \
            ["storycloze_2018", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge",
             "openbookqa", "boolq", "race_high", 'all_zs', 'all_rc']
    }
    output_0 = process_single_json(path_0) if os.path.isfile(path_0) else path_none
    output_1_1 = process_single_json(path_1_1) if os.path.isfile(path_1_1) else path_none
    output_1 = process_single_json(path_1) if os.path.isfile(path_1) else path_none

    for task in ["storycloze_2018", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]:
        print(f"{task}: {output_0[task]} | {output_1_1[task]} | {output_1[task]}")

    print(f"all: {output_0['all_zs']} | {output_1_1['all_zs']} | {output_1['all_zs']}")
    print("------------------------------")

    for task in ["boolq", "race_high"]:
        print(f"{task}: {output_0[task]} | {output_1_1[task]} | {output_1[task]}")

    print(f"all: {output_0['all_rc']} | {output_1_1['all_rc']} | {output_1['all_rc']}")


if __name__ == "__main__":
    log_file = sys.argv[1]
    if log_file.endswith(".json"):
        output = process_single_json(log_file)
        for task in ["storycloze_2018", "piqa", "hellaswag", "winogrande", "arc_easy",
                      "arc_challenge", "openbookqa", "all_zs",
                      "boolq", "race_high", "all_rc"]:
            print(f"{task}: {output[task]}")
            if task.startswith("all"):
                print("------------------------------")
    else:
        process_all(log_file + "_0.json", log_file + "_1-1.json", log_file + "_1.json")
