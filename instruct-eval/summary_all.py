import os
import sys
import argparse

def split_file(input_file, output_mark=None):
    if output_mark is None:
        output_mark = os.path.basename(input_file).replace(".txt", "")
    with open(input_file, 'r') as f:
        lines = f.readlines()

    mark = 0
    start_save_flag = False
    count = 0
    task_series = ["mmlu", "bbh", "mmlu", "bbh", "mmlu", "bbh"]

    for line in lines:
        # import pdb; pdb.set_trace()

        if line.startswith("prompt_mark "):
            mark = line.strip().replace("prompt_mark ", "")
            output_filename = f"./results/{output_mark}_{task_series[count]}_{mark}.txt"
            if os.path.isfile(output_filename):
                os.remove(output_filename)
            count += 1
            start_save_flag = True
            print(f"Created {output_filename}")

        if start_save_flag:
            with open(output_filename, 'a') as f:
                f.write(line)
    if count > 2:
        return output_mark, count
    else:
        return output_filename, count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_mark', type=str)
    args = parser.parse_args()

    output_mark, count = split_file(args.input_file, args.output_mark)
    if count > 2:
        os.system(f"python ./instruct-eval/summary_bbh.py ./results/{output_mark}_bbh")
        os.system(f"python ./instruct-eval/summary_mmlu.py ./results/{output_mark}_mmlu")
    else:
        os.system(f"python ./instruct-eval/summary_mmlu.py {output_mark.replace('bbh', 'mmlu')}")
        os.system(f"python ./instruct-eval/summary_bbh.py {output_mark}")
