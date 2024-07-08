import hydra
import torch
import os
import re
import cv2
import json
import argparse
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from mllm_npu.models.mllm import SEED
from mllm_npu.data.utils import process_anyres_image

import torch_npu
from torch_npu.contrib import transfer_to_npu


def main(args):
    BOI_TOKEN = '<img>'
    BOP_TOKEN = '<patch>'
    EOI_TOKEN = '</img>'
    EOP_TOKEN = '</patch>'
    IMG_TOKEN = '<img_{:05d}>'

    resolution_grids = ['1x1', '1x2', '1x3', '2x1', '3x1', '1x4', '4x1', '2x2']
    base_resolution = 448

    device = 'cuda:0'
    dtype = torch.float16
    dtype_str = 'fp16'
    num_img_in_tokens = 64
    num_img_out_tokens = 64

    model_cfg = OmegaConf.load(args.config_path).mllm

    language_model_cfg = model_cfg.language_model
    llm_model = hydra.utils.instantiate(language_model_cfg, torch_dtype=dtype_str)
    print("init llm done")

    tokenizer_cfg = model_cfg.tokenizer
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, add_prefix_space=False)
    print("init tokenizer done")

    visual_encoder_cfg = model_cfg.mllm_model.vision_encoder
    visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
    print("init visual encoder done")

    mllm_model_cfg = model_cfg.mllm_model
    mllm_model = hydra.utils.instantiate(mllm_model_cfg, language_model=llm_model)
    mllm_model.eval().to(device, dtype=dtype)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='img2txt inference demo')

    parser.add_argument('--config_path', type=str, default='./seed_npu/configs/models/mllm_llama2_13b_qwenvl_vit.yaml')
    parser.add_argument('--image_path', type=str, default='./images/img-1.png')
    parser.add_argument('--input_text', type=str, default='What is unusual about this image?')

    args = parser.parse_args()

    main(args)