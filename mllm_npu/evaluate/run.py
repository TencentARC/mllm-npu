import hydra
import torch
import re
import os
import time
import argparse
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from mllm_npu.models.mllm import SEED
from mllm_npu.data.utils import process_anyres_image

import torch_npu
from torch_npu.contrib import transfer_to_npu

from mllm_npu.evaluate.eval_data.mmlu import mmlu_eval
from mllm_npu.evaluate.eval_data.cmmlu import cmmlu_eval


def main(args):
    device = 'cuda:0'
    dtype = torch.bfloat16
    dtype_str = 'bf16'

    model_cfg = OmegaConf.load(args.config_path).mllm

    language_model_cfg = model_cfg.language_model
    llm_model = hydra.utils.instantiate(language_model_cfg, torch_dtype=dtype_str)
    print("init llm done")

    tokenizer_cfg = model_cfg.tokenizer
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, add_prefix_space=False)
    print("init tokenizer done")

    mllm_model_cfg = model_cfg.mllm_model
    mllm_model = hydra.utils.instantiate(mllm_model_cfg, language_model=llm_model)
    mllm_model.eval().to(device, dtype=dtype)
    print("init mllm done")

    if args.dataset_name == "mmlu":
        mmlu_eval(mllm_model, tokenizer, args.data_path, device)
    elif args.dataset_name == "cmmlu":
        cmmlu_eval(mllm_model, tokenizer, args.data_path, device)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate mllm on npu')

    parser.add_argument('--config_path', type=str, default='./mllm_npu/configs/models/seedx_llama2_13b_qwenvl_vitg.yaml')
    parser.add_argument('--dataset_name', type=str, default='mmlu')
    parser.add_argument('--data_path', type=str, default='./mllm_npu/evaluate/eval_data/mmlu/')

    args = parser.parse_args()

    main(args)
