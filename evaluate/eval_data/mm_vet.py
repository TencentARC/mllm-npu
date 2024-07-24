import os
import re
import json
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from PIL import Image
from mllm_npu.data.utils import process_anyres_image


def mm_vet_eval(model, tokenizer, image_transform, data_path, device):
    BOI_TOKEN = '<img>'
    BOP_TOKEN = '<patch>'
    EOI_TOKEN = '</img>'
    EOP_TOKEN = '</patch>'
    IMG_TOKEN = '<img_{:05d}>'

    resolution_grids = ['1x1', '1x2', '1x3', '2x1', '3x1', '1x4', '4x1', '2x2']
    base_resolution = 448

    device = 'cuda'
    dtype = torch.bfloat16
    dtype_str = 'bf16'
    num_img_in_tokens = 64
    num_img_out_tokens = 64

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]
    bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
    eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]

    grid_pinpoints = []
    for scale in resolution_grids:
        s1, s2 = scale.split('x')
        grid_pinpoints.append([int(s1) * base_resolution, int(s2) * base_resolution])
    grid_pinpoints = grid_pinpoints

    image_folder = os.path.join(data_path, "images")
    meta_data = os.path.join(data_path, "mm-vet.json")

    with open(meta_data, 'r') as f:
        data = json.load(f)

    results = {}

    for i in range(len(data)):
        id = f"v1_{i}"
        if id in results:
            continue

        imagename = data[id]['imagename']
        img_path = os.path.join(image_folder, imagename)
        image = Image.open(img_path).convert('RGB')
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
        image_tokens += BOI_TOKEN + ''.join(
            IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN

        prompt = image_tokens + 'You are a helpful assistant. Generate a short and concise response to the following image text pair.'.format(
            data[id]['question'])

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
            output = model.generate(
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

        results[id] = text

    with open("res_mmvet.json", 'w') as f:
        json.dump(results, f, indent=4)