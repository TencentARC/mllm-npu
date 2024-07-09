import hydra
import torch
import re
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

    mllm_model_cfg = model_cfg.mllm_model
    mllm_model = hydra.utils.instantiate(mllm_model_cfg, language_model=llm_model)
    mllm_model.eval().to(device, dtype=dtype)
    print("init mllm done")

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

    bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
    eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]

    grid_pinpoints = []
    for scale in resolution_grids:
        s1, s2 = scale.split('x')
        grid_pinpoints.append([int(s1) * base_resolution, int(s2) * base_resolution])
    grid_pinpoints = grid_pinpoints

    image_transform_cfg = model_cfg.processor
    image_transform_cfg["_target_"] = "mllm_npu.data.processor.image_processing_clip.CLIPImageProcessor"
    image_transform = hydra.utils.instantiate(image_transform_cfg)

    image = Image.open(args.image_path).convert('RGB')
    image_tensor, patch_pos_tensor = process_anyres_image(image, image_transform, grid_pinpoints, base_resolution)
    embeds_cmp_mask = torch.tensor([True] * image_tensor.shape[0]).to(device, dtype=torch.bool)

    patch_pos = [patch_pos_tensor]
    patch_position = torch.cat(patch_pos, dim=0)

    image_tensor = image_tensor.to(device, dtype=dtype)

    patch_length = image_tensor.shape[0]
    image_tokens = ''
    for _ in range(patch_length - 1):
        image_tokens += BOP_TOKEN + ''.join(
            IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOP_TOKEN
    image_tokens += BOI_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN

    prompt = image_tokens + 'Question: {}\nAnswer:'.format(args.input_text)

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + input_ids

    input_ids = torch.tensor(input_ids).to(device, dtype=torch.long)

    ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    boi_indices = torch.where(torch.logical_or(input_ids == boi_token_id, input_ids == bop_token_id))[0].tolist()
    eoi_indices = torch.where(torch.logical_or(input_ids == eoi_token_id, input_ids == eop_token_id))[0].tolist()

    for boi_idx, eoi_idx in zip(boi_indices, eoi_indices):
        ids_cmp_mask[boi_idx + 1:eoi_idx] = True

    input_ids = input_ids.unsqueeze(0)
    ids_cmp_mask = ids_cmp_mask.unsqueeze(0)

    with torch.no_grad():
        output = mllm_model.generate(
            tokenizer=tokenizer,
            input_ids=input_ids,
            pixel_values=image_tensor,
            embeds_cmp_mask=embeds_cmp_mask,
            patch_positions=patch_position,
            ids_cmp_mask=ids_cmp_mask,
            max_new_tokens=512,
            num_img_gen_tokens=num_img_out_tokens
        )

    text = re.sub('<[^>]*>', '', output['text'])
    text = re.sub(r'\[(.*)\]', '', text)
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='img2txt inference demo')

    parser.add_argument('--config_path', type=str, default='./mllm_npu/configs/models/seedx_llama2_13b_qwenvl_vitg.yaml')
    parser.add_argument('--image_path', type=str, default='./images/img-1.png')
    parser.add_argument('--input_text', type=str, default='What is unusual about this image?')

    args = parser.parse_args()

    main(args)
