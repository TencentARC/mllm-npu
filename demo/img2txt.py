import hydra
import torch
import os
import re
import cv2
import argparse
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

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

    print(args)

    models_cfg = OmegaConf.load(args.config_path)
    print(models_cfg)
    models = hydra.utils.instantiate(models_cfg)


    # tokenizer = hydra.utils.instantiate(models_cfg["mllm"]["tokenizer"])
    # print('Init tokenizer done')

    # visual_encoder = hydra.utils.instantiate(models_cfg["mllm"]["mllm_model"]["vision_encoder"])
    # print('Init visual encoder done')

    # llm = hydra.utils.instantiate(models_cfg["mllm"]["language_model"], torch_dtype=dtype_str)
    # print('Init llm done')

    # projector = hydra.utils.instantiate(models_cfg["mllm"]["mllm_model"]["input_resampler"])
    # print('Init projector done')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='img2txt inference demo')

    parser.add_argument('--config_path', type=str, default='./seed_npu/configs/models/seedx_llama2_13b_qwenvl_vit.yaml')
    parser.add_argument('--image_path', type=str, default='./images/img-1.png')
    parser.add_argument('--input_text', type=str, default='What is unusual about this image?')

    args = parser.parse_args()

    main(args)