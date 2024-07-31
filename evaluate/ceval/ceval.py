import os
import pandas as pd
import torch
import numpy as np
import json


choices = ["A", "B", "C", "D"]


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx]["question"]
    for j in range(4):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx][choices[j]])
    prompt += "\n答案:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx]["answer"])
    return prompt


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def gen_prompt(train_df, subject, k=-1):
    prompt = "以下是中国关于{}考试的单项选择题，请选出其中的正确答案。\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def eval(model, tokenizer, subject, dev_df, test_df, device):
    res_s = {}

    for i in range(test_df.shape[0]):
        k = 5
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

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

        res_s[str(i)] = output['text'][1]

    return res_s


def ceval_eval(model, tokenizer, data_path, device):
    k = 5
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_path, "test")) if ".csv" in f])
    all_result = {}

    for subject in subjects:
        print(subject)
        dev_df = pd.read_csv(os.path.join(data_path, "dev", subject + "_dev.csv"))
        test_df = pd.read_csv(os.path.join(data_path, "test", subject + "_test.csv"))

        result = eval(model, tokenizer, subject, dev_df, test_df, device)
        all_result[subject] = result

    json.dump(all_result, open("result_ceval.json", "w"))
