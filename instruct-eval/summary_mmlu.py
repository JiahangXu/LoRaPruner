with open("log.txt", "r") as fp:
    lines = fp.readlines()
    curr = []
    res = {}
    for line in lines:
        l = line.strip()
        if l == "":
            curr.append(res)
            res = {}
        else:
            if l != "------------------------------":
                if len(l.split(" ")) == 3:
                    res["all"] = l.split(" ")[2]
                else:
                    name, r = l.split(" ")[4], l.split(" ")[2]
                    res[name] = r
curr.append(res)
# print(curr)

all_score0, all_score1, all_score2 = 0, 0, 0
for task in ["humanities", "STEM", "social", "other", "all"]:
    print(f"{task}: {round(float(curr[0][task])*100,1)} | {round(float(curr[1][task])*100,1)} | {round(float(curr[2][task])*100,1)}")
