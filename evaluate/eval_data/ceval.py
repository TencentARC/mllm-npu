import os
import pandas as pd
import torch
import numpy as np


choices = ["A", "B", "C", "D"]


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\n答案:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
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
    cors = []

    # for i in range(test_df.shape[0]):
    for i in range(5):
        k = 5
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        label = test_df.iloc[i, test_df.shape[1] - 1]

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

        cor = output['text'][1] == label
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc


def ceval_eval(model, tokenizer, data_path, device):
    k = 5
    subjects = sorted([f.split(".csv")[0] for f in os.listdir(os.path.join(data_path, "val")) if ".csv" in f])

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(data_path, "dev", subject + ".csv"), header=None)[:k]
        test_df = pd.read_csv(os.path.join(data_path, "val", subject + ".csv"), header=None)

        dev_df.drop("id", axis=1, inplace=True)
        dev_df.drop("explanation", axis=1, inplace=True)
        test_df.drop("id", axis=1, inplace=True)

        cors, acc = eval(model, tokenizer, subject, dev_df, test_df, device)