import os
import json
import torch
import numpy as np


def format_example(data, idx, include_answer=True):
    prompt = "Question: {}\nAnswer:".format(data["examples"][idx]["input"])
    if include_answer:
        prompt += " {}\n\n".format(data["examples"][idx]["target"])
    return prompt


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def gen_prompt(train_data, subject, k=-1):
    prompt = "The following are questions (with answers) about {}.\n\n".format(format_subject(subject))
    for i in range(k):
        prompt += format_example(train_data, i)
    return prompt


def eval(model, tokenizer, subject, test_data, device):
    cors = []

    # for i in range(5, len(test_data["examples"])):
    for i in range(5, 10):
        k = 5
        prompt_end = format_example(test_data, i, include_answer=False)
        train_prompt = gen_prompt(test_data, subject, k)
        prompt = train_prompt + prompt_end
        label = test_data["examples"][i]["target"]

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = [tokenizer.bos_token_id] + input_ids
        input_ids = torch.tensor(input_ids).to(device, dtype=torch.long)
        input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            output = model.generate(
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=10
            )

        print("pred:", output['text'])
        print("label: ", label)
        # cor = output['text'][1] == label
        # cors.append(cor)

    # acc = np.mean(cors)
    # cors = np.array(cors)
    #
    # print("Average accuracy {:.3f} - {}".format(acc, subject))


def bbh_eval(model, tokenizer, data_path, device):
    k = 5
    subjects = sorted([f.split(".json")[0] for f in os.listdir(os.path.join(data_path, "data")) if ".json" in f])

    for subject in subjects:
        test_data = json.load(open(os.path.join(data_path, "data", subject + ".json"), "r"))

        eval(model, tokenizer, subject, test_data, device)
