import os
import json
import random

def load_data(dataset_name) -> list:
    """ read data from dataset file
    """
    file_path = f'./data/{dataset_name}_test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def generate_trigger(dataset_name, cot_trigger_no=1):
    if dataset_name == "aqua":
        direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif dataset_name == "gsm8k":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "commonsensqa":
        direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        # plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif dataset_name == "addsub":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "multiarith":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "strategyqa":
        direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif dataset_name == "svamp":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "singleeq":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "bigbench_date":
        direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif dataset_name == "object_tracking":
        direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif dataset_name == "coin_flip":
        direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif dataset_name == "last_letters":
        direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = direct_answer_trigger.replace("\nTherefore, ", "")
    direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    direct_answer_trigger_for_zeroshot_cot = direct_answer_trigger

    direct_answer_trigger_for_fewshot = "The answer is"

    if cot_trigger_no == 1:
        cot_trigger = "Let's think step by step."
    elif cot_trigger_no == 2:
        cot_trigger = "We should think about this step by step."
    elif cot_trigger_no == 3:
        cot_trigger = "First,"
    elif cot_trigger_no == 4:
        cot_trigger = "Before we dive into the answer,"
    elif cot_trigger_no == 5:
        cot_trigger = "Proof followed by the answer."
    elif cot_trigger_no == 6:
        cot_trigger = "Let's think step by step in a realistic way."
    elif cot_trigger_no == 7:
        cot_trigger = "Let's think step by step using common sense and knowledge."
    elif cot_trigger_no == 8:
        cot_trigger = "Let's think like a detective step by step."
    elif cot_trigger_no == 9:
        cot_trigger = "Let's think about this logically."
    elif cot_trigger_no == 10:
        cot_trigger = "Let's think step by step. First,"
    elif cot_trigger_no == 11:
        cot_trigger = "Let's think"
    elif cot_trigger_no == 12:
        cot_trigger = "Let's solve this problem by splitting it into steps."
    elif cot_trigger_no == 13:
        cot_trigger = "The answer is after the proof."
    elif cot_trigger_no == 14:
        cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    return (
        direct_answer_trigger_for_zeroshot,
        direct_answer_trigger_for_zeroshot_cot,
        cot_trigger,
        direct_answer_trigger_for_fewshot
    )


def create_demo_text(cot_flag, cot_length, direct_answer_trigger_for_fewshot="The answer is"):
    ''' dataset in ("multiarith", "gsm8k")
    '''
    x, z, y = [], [], [] # x: instruction; z: output; y: answer

    x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
    z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
    # z.append("21 - 15 = 6.") # shorter version
    y.append("6")

    x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
    z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    # z.append("3 + 2 = 5.")
    y.append("5")        

    x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
    z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
    # z.append("In total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
    y.append("39")        

    x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
    z.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
    # z.append("Jason gave Denny 20 - 12 = 8.")
    y.append("8")        

    x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
    z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
    # z.append("2 toys each from his mom and dad is 4 more toys. 5 + 4 = 9.")
    y.append("9")        

    x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
    z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
    # z.append("5 * 4 = 20 computers were added. 9 + 20 is 29.")
    y.append("29")        

    x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
    z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
    # z.append("After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33.")
    y.append("33")        

    x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
    z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
    z.append("5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8.")
    y.append("8")

    # # randomize order of the examples ...
    # random.seed(42)
    index_list = list(range(cot_length))
    # random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text


