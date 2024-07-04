import os
import functools

import torch
from PIL import Image
import torchdata.datapipes as dp
from braceexpand import braceexpand

from mllm_npu.data.data_utils import (filter_data_with_image_ids,
                                      llava_collate)
from mllm_npu.constant import (BOI_TOKEN, EOI_TOKEN, BOP_TOKEN, EOP_TOKEN,
                               IMG_TOKEN, dynamic_padding)

from mllm_npu.data.utils import (process_anyres_image, anyres_data_collate,
                                 anyres_data_collate_old)


def decode_llava_data_caption(item,
                              image_dir,
                              tokenizer,
                              image_transform=None,
                              caption_prompt='',
                              max_length=128,
                              min_resolution=400,
                              ratio_index=0,
                              min_aspect_ratio=0.666,
                              max_aspect_ratio=10.0,
                              num_img_in_tokens=64,
                              num_img_out_tokens=64,
                              multi_resolution=False,
                              resolution_grids=None,
                              base_resolution=224,
                              grid_pinpoints=None):

    key, value = item

    if value.get('data', None) is None:
        return {}

    if 'image' in value:
        image_path = os.path.join(image_dir, value['image'])

        try:
            image = Image.open(image_path).convert('RGB')

            if multi_resolution:
                img_size = image.size
                image, patch_pos = process_anyres_image(
                    image, image_transform, grid_pinpoints, base_resolution)
                images_patch_length = torch.tensor([len(patch_pos)],
                                                   dtype=torch.long)
                image_size = torch.tensor([img_size], dtype=torch.long)
                embeds_gen_mask = [False] * len(patch_pos)
                embeds_cmp_mask = [True] * len(patch_pos)

            else:

                image = image_transform(image)

                embeds_gen_mask = False
                embeds_cmp_mask = True
        except Exception as e:
            print('Error while decode image: ', e)
            return {}
    else:
        image = None
        embeds_gen_mask = None
        embeds_cmp_mask = None

    input_ids = []
    labels = []
    input_text = ''

    for idx, content in enumerate(value['data']):
        if idx % 2 == 0:
            if image is not None:
                if multi_resolution:
                    image_tokens = ''
                    for patch_legnth in images_patch_length.tolist():
                        for _ in range(patch_legnth - 1):
                            image_tokens += BOP_TOKEN + ''.join([
                                IMG_TOKEN.format(int(item))
                                for item in range(num_img_in_tokens)
                            ]) + EOP_TOKEN
                        image_tokens += BOI_TOKEN + ''.join([
                            IMG_TOKEN.format(int(item))
                            for item in range(num_img_in_tokens)
                        ]) + EOI_TOKEN

                else:
                    image_tokens = BOI_TOKEN + ''.join([
                        IMG_TOKEN.format(int(item))
                        for item in range(num_img_in_tokens)
                    ]) + EOI_TOKEN
            else:
                image_tokens = ''

            text = image_tokens + caption_prompt
            item_ids = tokenizer.encode(text, add_special_tokens=False)
            item_labels = [-100] * len(item_ids)

        else:
            text = content
            item_ids = tokenizer.encode(text, add_special_tokens=False)
            item_labels = item_ids

            has_large_element = any(x >= tokenizer.vocab_size
                                    for x in item_ids)
            if has_large_element:
                print(text)
                return {}

        input_text += text
        input_ids.extend(item_ids)
        labels.extend(item_labels)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] + labels + [tokenizer.eos_token_id]

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]
    ids_cmp_mask = [False] * len(input_ids)
    ids_gen_mask = [False] * len(input_ids)

    if image is not None:
        boi_idx = input_ids.index(boi_token_id)
        eoi_idx = input_ids.index(eoi_token_id)

        if eoi_idx >= max_length:
            print("max length exceeded")
            return {}

    if len(input_ids) >= max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        ids_cmp_mask = ids_cmp_mask[:max_length]
        ids_gen_mask = ids_gen_mask[:max_length]
    elif not dynamic_padding:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
    embeds_cmp_mask = torch.tensor(
        embeds_cmp_mask) if embeds_cmp_mask is not None else None
    embeds_gen_mask = torch.tensor(
        embeds_gen_mask) if embeds_gen_mask is not None else None

    if image is not None:
        ids_cmp_mask[boi_idx + 1:eoi_idx] = True

    if multi_resolution:
        bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
        eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]
        bop_indices = torch.where(input_ids == bop_token_id)
        eop_indices = torch.where(input_ids == eop_token_id)

        for bop_idx, eop_idx in zip(bop_indices[0], eop_indices[0]):
            ids_cmp_mask[bop_idx + 1:eop_idx] = True

    ret = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        'images': image,
        'text': input_text,
    }

    if multi_resolution:
        ret.update({
            'images_patch_length': images_patch_length,
            'patch_position': patch_pos,
            'image_size': image_size,
        })

    return ret


def build_ocr_datapipes_caption(data_dir,
                                image_dir,
                                tokenizer=None,
                                max_length=77,
                                batch_size=None,
                                min_resolution=180,
                                ratio_index=0,
                                image_transform=None,
                                caption_prompt='',
                                min_aspect_ratio=0.666,
                                max_aspect_ratio=10.0,
                                num_img_in_tokens=64,
                                num_img_out_tokens=64,
                                cycle_count=None,
                                multi_resolution=False,
                                resolution_grids=None,
                                base_resolution=224,
                                dataset_name=None):
    """
    datapipe of ocr dataset (such as LLaVAR, Slides...) with jsonl format
    """
    grid_pinpoints = []
    if multi_resolution:
        resolution_grids = list(resolution_grids)

        for scale in resolution_grids:
            s1, s2 = scale.split('x')
            grid_pinpoints.append(
                [int(s1) * base_resolution,
                 int(s2) * base_resolution])

    decode_partial = functools.partial(decode_llava_data_caption,
                                       image_dir=image_dir,
                                       tokenizer=tokenizer,
                                       image_transform=image_transform,
                                       caption_prompt=caption_prompt,
                                       max_length=max_length,
                                       min_resolution=min_resolution,
                                       ratio_index=ratio_index,
                                       min_aspect_ratio=min_aspect_ratio,
                                       max_aspect_ratio=max_aspect_ratio,
                                       num_img_in_tokens=num_img_in_tokens,
                                       num_img_out_tokens=num_img_out_tokens,
                                       multi_resolution=multi_resolution,
                                       resolution_grids=resolution_grids,
                                       base_resolution=base_resolution,
                                       grid_pinpoints=grid_pinpoints)

    filter_partial = functools.partial(filter_data_with_image_ids)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))

    datapipe = dp.iter.FileLister(root=data_dir,
                                  masks='*.jsonl',
                                  recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(filter_partial)

    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        if dynamic_padding:
            collate_func = functools.partial(anyres_data_collate,
                                             tokenizer=tokenizer,
                                             dataset_name=dataset_name)
        else:
            collate_func = functools.partial(anyres_data_collate_old,
                                             dataset_name=dataset_name)
        datapipe = datapipe.collate(
            collate_fn=collate_func if multi_resolution else llava_collate)

    return datapipe
