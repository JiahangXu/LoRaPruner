import os
import sys


def parse_one_file(name):
    start_save_flag = False
    with open(name, "r", encoding='utf-8') as fp:
        lines = fp.readlines()
    res = {}
    count = 0
    for line in lines:
        l = line.strip()
        if start_save_flag and count < 5:
            if len(l.split(" ")) == 3:
                res["all"] = l.split(" ")[2]
            else:
                name, r = l.split(" ")[4], l.split(" ")[2]
                res[name] = r
            count += 1
        if l == "------------":
            start_save_flag = True
    return res

def parse_all(path_0=None, path_1_1=None, path_1=None):
    path_none = {k: "" for k in ["humanities", "STEM", "social", "other", "all"]}
    output_0 = parse_one_file(path_0) if os.path.isfile(path_0) else path_none
    output_1_1 = parse_one_file(path_1_1) if os.path.isfile(path_1_1) else path_none
    output_1 = parse_one_file(path_1) if os.path.isfile(path_1) else path_none

    for task in ["humanities", "STEM", "social", "other", "all"]:
    #     print(f"{task}: {round(float(curr[0][task])*100,1)} | {round(float(curr[1][task])*100,1)} | {round(float(curr[2][task])*100,1)}")

    # for task in ["NLP", "Algorithms", "Total"]:
        print(f"{task}: {round(float(output_0[task])*100,1)} | {round(float(output_1_1[task])*100,1)} | {round(float(output_1[task])*100,1)}")


if __name__ == "__main__":
    log_file = sys.argv[1]
    print("----------", log_file)
    if log_file.endswith(".txt"):
        output = parse_one_file(log_file)
        for task in ["humanities", "STEM", "social", "other", "all"]:
            print(f"{task}: {round(float(output[task])*100,1)}")
    else:
        parse_all(log_file + "_0.txt", log_file + "_1-1.txt", log_file + "_1.txt")






