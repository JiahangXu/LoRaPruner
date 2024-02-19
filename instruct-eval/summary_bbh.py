import os
import sys

all = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

def get_categories():
    return {
        "NLP": [
            'disambiguation_qa',
            'hyperbaton',
            'salient_translation_error_detection',
            'snarks',
            'sports_understanding',
            'movie_recommendation',
            'date_understanding',
            'causal_judgement',
            'ruin_names',
            'formal_fallacies',
            'penguins_in_a_table',
            'reasoning_about_colored_objects',
        ],
        "Algorithm": [
            'multistep_arithmetic_two',
            'boolean_expressions',
            'logical_deduction_five_objects',
            'logical_deduction_seven_objects',
            'logical_deduction_three_objects',
            'geometric_shapes',
            'dyck_languages',
            'navigate',
            'temporal_sequences',
            'tracking_shuffled_objects_five_objects',
            'tracking_shuffled_objects_seven_objects',
            'tracking_shuffled_objects_three_objects',
            'object_counting',
            'web_of_lies',
            'word_sorting',
        ]
    }

def parse_one_file(name):
    all_results = []

    with open(name, "r", encoding='utf-8') as fp:
        lines = fp.readlines()

        for line in lines:
            l = line.strip()
            if "{'name': '" in l:
                all_results.append(eval(l))

    names = []
    for item in all_results:
        if item['name'] not in names:
            names.append(item['name'])
        else:
            print("error!!!!", item['name'])
        # print(f"{item['name']}:{item['score']}")
    for item in all:
        if item not in names:
            print("missing item:", item)

    nlp, algorithms = [], []
    for item in all_results:
        if "name" in item and item["name"] in get_categories()["NLP"]:
            nlp.append(item["score"])
        else:
            algorithms.append(item["score"])

    # print(len(nlp), len(algorithms))

    score = sum(res["score"] for res in all_results) / len(all_results)

    assert len(nlp) == 12 and len(algorithms) == 15
    # print("NLP", round(sum(nlp) / len(nlp) * 100, 2))
    # print("Algorithms", round(sum(algorithms) / len(algorithms) * 100, 2))
    # print("Total", round(score * 100, 2))
    return {
        "NLP": round(sum(nlp) / len(nlp) * 100, 2),
        "Algorithms": round(sum(algorithms) / len(algorithms) * 100, 2),
        "Total": round(score * 100, 2)
    }

def parse_all(path_0=None, path_1_1=None, path_1=None):
    path_none = {k: "" for k in ["NLP", "Algorithms", "Total"]}
    output_0 = parse_one_file(path_0) if os.path.isfile(path_0) else path_none
    output_1_1 = parse_one_file(path_1_1) if os.path.isfile(path_1_1) else path_none
    output_1 = parse_one_file(path_1) if os.path.isfile(path_1) else path_none

    for task in ["NLP", "Algorithms", "Total"]:
        print(f"{task}: {round(float(output_0[task]),2)} | {round(float(output_1_1[task]),2)} | {round(float(output_1[task]),2)}")


if __name__ == "__main__":
    log_file = sys.argv[1]
    print("----------", log_file)
    if log_file.endswith(".txt"):
        output = parse_one_file(log_file)
        for task in ["NLP", "Algorithms", "Total"]:
            print(f"{task}: {round(float(output[task]),2)}")
    else:
        parse_all(log_file + "_0.txt", log_file + "_1-1.txt", log_file + "_1.txt")
