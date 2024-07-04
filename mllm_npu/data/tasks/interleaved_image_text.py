import pickle
import functools

import torch
import numpy as np
import torchdata.datapipes as dp
from braceexpand import braceexpand

from mllm_npu.data.data_utils import (select, unwarp_data,
                                      filter_data_with_image_ids,
                                      base64_to_image, mmc4_collate)
from mllm_npu.constant import (BOI_TOKEN, EOI_TOKEN, BOP_TOKEN, EOP_TOKEN,
                               IMG_TOKEN, dynamic_padding)

from mllm_npu.data.utils import (process_anyres_image, anyres_data_collate,
                                 anyres_data_collate_old)


def decode_interleave_data(item,
                           tokenizer=None,
                           image_transform=None,
                           max_length=1024,
                           img_first_ratio=1.0,
                           num_img_in_tokens=64,
                           num_img_out_tokens=64,
                           multi_resolution=False,
                           resolution_grids=None,
                           base_resolution=224,
                           grid_pinpoints=None):
    key, value = item
    sample = {}
    if key.endswith(".pkl"):
        try:
            value = pickle.load(value)

            image_list = value['images']
            text_list = value['texts']

            images = []
            input_ids = [tokenizer.bos_token_id]
            labels = [-100]

            cur_seq_length = 1
            embeds_cmp_mask = []
            embeds_gen_mask = []
            ids_cmp_mask = [False]
            ids_gen_mask = [False]
            images_patch_length = []
            image_size = []
            patch_position = []
            input_text = ''
            for image_base64, text in zip(image_list, text_list):
                assert (image_base64 is None) + (text is None) == 1
                if image_base64 is not None:
                    image = base64_to_image(image_base64)
                    img_first_flag = np.random.uniform(0, 1) < img_first_ratio

                    if img_first_flag:

                        if multi_resolution:
                            img_size = image.size
                            image, patch_pos = process_anyres_image(
                                image, image_transform, grid_pinpoints,
                                base_resolution)

                            if cur_seq_length + (num_img_in_tokens + 2) * len(
                                    patch_pos) >= max_length:
                                break
                            patch_position.append(patch_pos)
                            images_patch_length.append(len(patch_pos))
                            image_size.append(img_size)

                            embeds_cmp_mask.extend([True] * len(patch_pos))
                            embeds_gen_mask.extend([False] * len(patch_pos))
                            image_tokens = ''
                            for _ in range(len(patch_pos) - 1):
                                image_tokens += BOP_TOKEN + ''.join([
                                    IMG_TOKEN.format(int(item))
                                    for item in range(num_img_in_tokens)
                                ]) + EOP_TOKEN
                            image_tokens += BOI_TOKEN + ''.join([
                                IMG_TOKEN.format(int(item))
                                for item in range(num_img_in_tokens)
                            ]) + EOI_TOKEN

                            image_ids = tokenizer.encode(
                                image_tokens, add_special_tokens=False)
                            image_labels = [-100] * len(image_ids)

                            for i in range(len(patch_pos)):
                                ids_cmp_mask.extend([False] + [True] *
                                                    num_img_in_tokens +
                                                    [False])
                                ids_gen_mask.extend([False] + [False] *
                                                    num_img_in_tokens +
                                                    [False])

                        else:

                            if cur_seq_length + num_img_in_tokens >= max_length:
                                break
                            embeds_cmp_mask.append(True)
                            embeds_gen_mask.append(False)
                            image_tokens = BOI_TOKEN + ''.join([
                                IMG_TOKEN.format(int(item))
                                for item in range(num_img_in_tokens)
                            ]) + EOI_TOKEN
                            image_ids = tokenizer.encode(
                                image_tokens, add_special_tokens=False)
                            image_labels = [-100] * len(image_ids)
                            ids_cmp_mask.extend([False] +
                                                [True] * num_img_in_tokens +
                                                [False])
                            ids_gen_mask.extend([False] +
                                                [False] * num_img_in_tokens +
                                                [False])

                    else:
                        if cur_seq_length + num_img_out_tokens >= max_length:
                            break
                        embeds_cmp_mask.append(False)
                        embeds_gen_mask.append(True)
                        image_tokens = BOI_TOKEN + ''.join([
                            IMG_TOKEN.format(int(item))
                            for item in range(num_img_out_tokens)
                        ]) + EOI_TOKEN
                        image_ids = tokenizer.encode(image_tokens,
                                                     add_special_tokens=False)
                        image_labels = [image_ids[0]
                                        ] + [-100] * (len(image_ids) - 1)
                        ids_cmp_mask.extend([False] +
                                            [False] * num_img_out_tokens +
                                            [False])
                        ids_gen_mask.extend([False] +
                                            [True] * num_img_out_tokens +
                                            [False])

                    if image_transform is not None and not multi_resolution:
                        image = image_transform(image)

                    images.append(image)

                    input_ids.extend(image_ids)
                    labels.extend(image_labels)
                    cur_seq_length += len(image_ids)
                    input_text += image_tokens

                else:
                    text_ids = tokenizer.encode(text, add_special_tokens=False)
                    if cur_seq_length + len(text_ids) >= max_length:
                        break
                    input_ids.extend(text_ids)
                    ids_cmp_mask.extend([False] * len(text_ids))
                    ids_gen_mask.extend([False] * len(text_ids))
                    labels.extend(text_ids)
                    cur_seq_length += len(text_ids)
                    input_text += text

            input_ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)
            ids_cmp_mask.append(False)
            ids_gen_mask.append(False)
            attention_mask = [1] * len(input_ids)
            if len(input_ids) >= max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
                ids_gen_mask = ids_gen_mask[:max_length]
                ids_cmp_mask = ids_cmp_mask[:max_length]

            elif not dynamic_padding:
                padding_length = max_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id
                                         ] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length
                ids_gen_mask = ids_gen_mask + [False] * padding_length
                ids_cmp_mask = ids_cmp_mask + [False] * padding_length

            if multi_resolution:
                if len(images) > 0:
                    images = torch.cat(images)

            assert len(images) == len(embeds_cmp_mask) and len(images) == len(
                embeds_gen_mask)
            if image_transform is not None and len(
                    images) > 0 and not multi_resolution:
                images = torch.stack(images)

            if len(images) == 0:
                return key, {}
                images = None
                embeds_cmp_mask = None
                embeds_gen_mask = None
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
            ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
            embeds_gen_mask = torch.tensor(
                embeds_gen_mask,
                dtype=torch.bool) if embeds_gen_mask is not None else None
            embeds_cmp_mask = torch.tensor(
                embeds_cmp_mask,
                dtype=torch.bool) if embeds_cmp_mask is not None else None

            ret = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'ids_gen_mask': ids_gen_mask,
                'images': images,
                'ids_cmp_mask': ids_cmp_mask,
                'embeds_gen_mask': embeds_gen_mask,
                'embeds_cmp_mask': embeds_cmp_mask,
                'text': input_text
            }
            if multi_resolution:
                ret.update({
                    'images_patch_length':
                    torch.tensor(images_patch_length, dtype=torch.long),
                    'patch_position':
                    torch.cat(patch_position, dim=0),
                    'image_size':
                    torch.tensor(image_size, dtype=torch.long),
                })
            return key, ret
        except Exception as e:
            print(f'Error occured when decode: {e}')
            return key, {}
    else:
        return key, {}


def build_interleave_datapipes_with_pixels(data_dir,
                                           tokenizer=None,
                                           max_length=1024,
                                           batch_size=None,
                                           image_transform=None,
                                           img_first_ratio=0.5,
                                           num_img_in_tokens=64,
                                           num_img_out_tokens=64,
                                           cycle_count=None,
                                           multi_resolution=False,
                                           resolution_grids=None,
                                           base_resolution=224,
                                           dataset_name=None):
    """
    datapipe of image text interleave dataset (such as MMC4, OBELISC...) with webdataset format
    """
    grid_pinpoints = []
    if multi_resolution:
        resolution_grids = list(resolution_grids)

        for scale in resolution_grids:
            s1, s2 = scale.split('x')
            grid_pinpoints.append(
                [int(s1) * base_resolution,
                 int(s2) * base_resolution])

    decode_partial = functools.partial(decode_interleave_data,
                                       tokenizer=tokenizer,
                                       image_transform=image_transform,
                                       max_length=max_length,
                                       img_first_ratio=img_first_ratio,
                                       num_img_in_tokens=num_img_in_tokens,
                                       num_img_out_tokens=num_img_out_tokens,
                                       multi_resolution=multi_resolution,
                                       resolution_grids=resolution_grids,
                                       base_resolution=base_resolution,
                                       grid_pinpoints=grid_pinpoints)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))

    datapipe = dp.iter.FileLister(root=data_dir, masks='*.tar', recursive=True)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    datapipe = datapipe.load_from_tar_wo_exception()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.webdataset()
    datapipe = datapipe.map(unwarp_data)
    datapipe = datapipe.filter(filter_data_with_image_ids)
    datapipe = datapipe.map(select)

    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        if multi_resolution:
            if dynamic_padding:
                collate_func = functools.partial(anyres_data_collate,
                                                 tokenizer=tokenizer,
                                                 dataset_name=dataset_name)
            else:
                collate_func = functools.partial(anyres_data_collate_old,
                                                 dataset_name=dataset_name)
            datapipe = datapipe.collate(collate_fn=collate_func)
        else:
            datapipe = datapipe.collate(mmc4_collate)

    return datapipe
