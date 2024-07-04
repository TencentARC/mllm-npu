import json
import random
import functools
import pyrootutils

import torch
import numpy as np
from PIL import Image
import torchdata.datapipes as dp
from braceexpand import braceexpand

from mllm_npu.data.data_utils import (select, filter_data_with_similarity,
                                      unwarp_data)
from mllm_npu.constant import (BOI_TOKEN, EOI_TOKEN, BOP_TOKEN, EOP_TOKEN,
                               IMG_TOKEN, dynamic_padding)

from mllm_npu.data.utils import (process_anyres_image, anyres_data_collate,
                                 anyres_data_collate_old)

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

double_image_prompt = [
    'Produce the identical picture',
    'Create the same visual',
    'Generate an identical image',
    'Reproduce the same picture',
    'Formulate the same visual',
    'Construct the same image',
    'Develop the identical picture',
    'Generate a similar image',
    'Create a matching picture',
    'Fabricate the same visual.',
    'Yield a comparable illustration',
    'Render a parallel graphic',
    'Synthesize a corresponding photo',
    'Generate a like representation',
    'Establish a related depiction',
    'Create an akin visual element',
    'Compose an analogous design',
    'Produce a coinciding portrayal',
    'Bring forth a consistent icon',
    'Form a congruent image',
]

gen_prompt_all = [
    "Please show me a picture of",
    "Please design an image of",
    "Please produce a photo of",
    "Please generate an image of",
    "Please draw a painting of",
    "I'd like to see a drawing of",
    "I'd love to see an illustration of",
    "I'd like to view an image of",
    "I want to see a picture of",
    "I would like to see a photo of",
    "Show me a photo of",
    "Generate a picture of",
    "Show me a photograph of",
    "Generate an image of",
    "Generate an image:",
    "Generate a picture:",
    "Generate a painting:",
    "Generate a photograph:",
    "Show me a photograph:",
    "Draw a picture:",
    "Draw a painting:",
    "Draw an image:",
    "Can you make an image of",
    "Can you draw a painting of",
    "Can you produce a picture of",
    "Can you generate a photo of",
    "Can you depict a picture of",
    "Can you show me an illustration of",
]

gen_prompt_response_all = [
    "Here is a picture.",
    "I have designed an image.",
    "Here is a photo.",
    "I have generated an image.",
    "Here's a painting.",
    "Here's a drawing.",
    "Enjoy this illustration.",
    "Take a look at this image.",
    "Here is a picture.",
    "I have created a photo.",
    "Enjoy this photo.",
    "I have generated a picture.",
    "Here is a photograph.",
    "Here's an image.",
    "Here's an image.",
    "Here's a picture.",
    "Here's a painting.",
    "Here's a photograph.",
    "Here's a photograph.",
    "Enjoy this picture.",
    "Enjoy this painting.",
    "Enjoy this image.",
    "Absolutely, here is an image.",
    "Absolutely, here is a painting.",
    "Sure, here is a picture.",
    "Of course, here is a photo.",
    "Certainly, please enjoy this picture.",
    "Sure, please enjoy this illustration.",
]


def tokenize_text(data,
                  tokenizer,
                  turn_sep='\n',
                  img_first_ratio=0.5,
                  max_length=128,
                  num_img_in_tokens=64,
                  num_img_out_tokens=64):

    if 'images' not in data:
        images_patch_length = [1]
    elif 'images_patch_length' in data:
        images_patch_length = data['images_patch_length'].tolist()
    else:
        raise NotImplementedError(' not supported yet.')

    if len(images_patch_length) == 1:
        patches_tokens = images_patch_length[0]
        if 'text' in data:
            caption = data['text']
            response = data['response']
            if patches_tokens * (num_img_in_tokens + 2) + 2 > max_length:
                data.pop('text')
                print('An example with patches tokens', patches_tokens,
                      'exceeds max length', max_length)
                return data
            results = encode_caption_input_ids_v2(
                caption=caption,
                response=response,
                tokenizer=tokenizer,
                turn_sep=turn_sep,
                img_first_ratio=img_first_ratio,
                max_length=max_length,
                num_img_in_tokens=num_img_in_tokens,
                num_img_out_tokens=num_img_out_tokens,
                patch_length=patches_tokens)

            if len(results.get('input_ids', [])) == 0:
                data.pop('text')
                return data

            data.update({
                'input_ids': results['input_ids'],
                'attention_mask': results['attention_mask'],
                'labels': results['labels'],
                'ids_gen_mask': results['ids_gen_mask'],
                'ids_cmp_mask': results['ids_cmp_mask'],
                'embeds_gen_mask': results['embeds_gen_mask'],
                'embeds_cmp_mask': results['embeds_cmp_mask'],
                'text': caption
            })

    else:
        raise NotImplementedError(
            'Multi-resolution for multi-images in a sequence is not supported yet.'
        )

    return data


def encode_caption_input_ids(caption,
                             tokenizer,
                             img_first_ratio,
                             max_length,
                             num_img_in_tokens=64,
                             num_img_out_tokens=64):
    """
    encapsulating data
    """
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)
    caption_labels = caption_ids

    img_first_flag = np.random.uniform(0, 1) < img_first_ratio

    if len(caption_ids) + num_img_out_tokens + 4 > max_length:
        img_first_flag = True

    if img_first_flag:
        image_tokens = BOI_TOKEN + ''.join(
            [IMG_TOKEN.format(int(item))
             for item in range(num_img_in_tokens)]) + EOI_TOKEN

        image_ids = tokenizer.encode(image_tokens, add_special_tokens=False)
        image_labels = [-100] * len(image_ids)

        input_ids = [tokenizer.bos_token_id
                     ] + image_ids + caption_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100
                  ] + image_labels + caption_labels + [tokenizer.eos_token_id]
        ids_gen_mask = [False] * len(input_ids)
        ids_cmp_mask = [False] + [False] + [True] * num_img_in_tokens + [
            False
        ] + [False] * len(caption_ids) + [False]
        embeds_gen_mask = False
        embeds_cmp_mask = True
    else:
        image_tokens = BOI_TOKEN + ''.join([
            IMG_TOKEN.format(int(item)) for item in range(num_img_out_tokens)
        ]) + EOI_TOKEN

        image_ids = tokenizer.encode(image_tokens, add_special_tokens=False)
        image_labels = [image_ids[0]] + [-100] * (len(image_ids) - 1)

        input_ids = [tokenizer.bos_token_id
                     ] + caption_ids + image_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * (len(caption_ids) + 1) + image_labels + [
            tokenizer.eos_token_id
        ]
        ids_gen_mask = [False] + [False] * len(caption_ids) + [
            False
        ] + [True] * num_img_out_tokens + [False] + [False]
        ids_cmp_mask = [False] * len(input_ids)
        embeds_gen_mask = True
        embeds_cmp_mask = False

    if len(input_ids) >= max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        ids_gen_mask = ids_gen_mask[:max_length]
        ids_cmp_mask = ids_cmp_mask[:max_length]

    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
    embeds_gen_mask = torch.tensor(embeds_gen_mask, dtype=torch.bool)
    embeds_cmp_mask = torch.tensor(embeds_cmp_mask, dtype=torch.bool)
    # print('encode_v1:', input_ids.shape, max_length)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
    }


def encode_caption_input_ids_v2(caption,
                                response,
                                tokenizer,
                                turn_sep,
                                img_first_ratio,
                                max_length,
                                num_img_in_tokens=64,
                                num_img_out_tokens=64,
                                patch_length=1):

    caption_ids = tokenizer.encode(caption, add_special_tokens=False)
    caption_labels = [-100] * len(caption_ids)

    response_ids = tokenizer.encode(response, add_special_tokens=False)
    response_labels = response_ids

    img_first_flag = np.random.uniform(0, 1) < img_first_ratio

    if img_first_flag:

        image_tokens = ''
        for i in range(patch_length - 1):
            image_tokens += BOP_TOKEN + ''.join([
                IMG_TOKEN.format(int(item))
                for item in range(num_img_in_tokens)
            ]) + EOP_TOKEN

        image_tokens += BOI_TOKEN + ''.join(
            [IMG_TOKEN.format(int(item))
             for item in range(num_img_in_tokens)]) + EOI_TOKEN

        image_ids = tokenizer.encode(image_tokens, add_special_tokens=False)
        image_labels = [-100] * len(image_ids)

        input_ids = [tokenizer.bos_token_id
                     ] + image_ids + caption_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        # assign caption_ids to labels
        caption_labels = caption_ids
        labels = [-100
                  ] + image_labels + caption_labels + [tokenizer.eos_token_id]
        ids_gen_mask = [False] * len(input_ids)

        ids_cmp_mask = [False]
        for i in range(patch_length):
            ids_cmp_mask += [False] + [True] * num_img_in_tokens + [False]

        ids_cmp_mask += [False] * len(caption_ids) + [False]

        embeds_gen_mask = [False] * patch_length
        embeds_cmp_mask = [True] * patch_length

    else:

        image_tokens = BOI_TOKEN + ''.join([
            IMG_TOKEN.format(int(item)) for item in range(num_img_out_tokens)
        ]) + EOI_TOKEN

        image_ids = tokenizer.encode(image_tokens, add_special_tokens=False)
        image_labels = [image_ids[0]] + [-100] * (len(image_ids) - 1)

        sep_ids = tokenizer.encode(turn_sep, add_special_tokens=False)
        sep_labels = sep_ids

        input_ids = [tokenizer.bos_token_id
                     ] + caption_ids + response_ids + image_ids + sep_ids + [
                         tokenizer.eos_token_id
                     ]
        attention_mask = [1] * len(input_ids)
        labels = [
            -100
        ] + caption_labels + response_labels + image_labels + sep_labels + [
            tokenizer.eos_token_id
        ]
        ids_gen_mask = [
            False
        ] + [False] * len(caption_ids) + [False] * len(response_ids) + [
            False
        ] + [True] * num_img_out_tokens + [False] + [False] + [False]
        ids_cmp_mask = [False] * len(input_ids)

        embeds_gen_mask = [False] * (patch_length - 1) + [True]
        embeds_cmp_mask = [False] * patch_length

    if len(input_ids) >= max_length:
        return {}

    elif not dynamic_padding:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
    embeds_gen_mask = torch.tensor(embeds_gen_mask, dtype=torch.bool)
    embeds_cmp_mask = torch.tensor(embeds_cmp_mask, dtype=torch.bool)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
    }


def decode_image_text_pair(item,
                           tokenizer,
                           image_transform=None,
                           max_length=128,
                           use_caption_in_metadata=False,
                           instruction_prompt=None,
                           add_gen_prompt=False,
                           caption_key_in_metadata='',
                           min_resolution=400,
                           min_aspect_ratio=0.666,
                           img_first_ratio=0.5,
                           num_img_in_tokens=64,
                           num_img_out_tokens=64,
                           assure_text=True,
                           multi_resolution=False,
                           resolution_grids=None,
                           base_resolution=224,
                           grid_pinpoints=None):
    """
    process text, image, json data in wds
    """
    key, value = item

    if key.endswith(".txt") and assure_text:
        if not use_caption_in_metadata:
            caption = value.read().decode('utf-8')
            if add_gen_prompt:
                num_ids = random.randint(0, len(gen_prompt_all) - 1)
                gen_prompt = gen_prompt_all[num_ids]
                gen_prompt_response = gen_prompt_response_all[num_ids]
            else:
                gen_prompt = None
                gen_prompt_response = None

            if gen_prompt is not None:
                caption = gen_prompt + ' ' + caption.lstrip(' ')

            if instruction_prompt is not None:
                caption = instruction_prompt.format_map(
                    {'instruction': caption})

            response = ''
            if gen_prompt_response is not None:
                response = gen_prompt_response

            if tokenizer is None or multi_resolution:
                return key, {'text': caption, 'response': response}
            else:

                results = encode_caption_input_ids(
                    caption=caption,
                    tokenizer=tokenizer,
                    img_first_ratio=img_first_ratio,
                    max_length=max_length,
                    num_img_in_tokens=num_img_in_tokens,
                    num_img_out_tokens=num_img_out_tokens)

                return key, {
                    'input_ids': results['input_ids'],
                    'attention_mask': results['attention_mask'],
                    'labels': results['labels'],
                    'ids_gen_mask': results['ids_gen_mask'],
                    'ids_cmp_mask': results['ids_cmp_mask'],
                    'embeds_gen_mask': results['embeds_gen_mask'],
                    'embeds_cmp_mask': results['embeds_cmp_mask'],
                    'text': caption
                }
        else:
            return key, {}
    elif key.endswith(".jpg"):
        try:
            image = Image.open(value).convert('RGB')
            width, height = image.size
        except Exception as e:
            print('Error while decode image: ', e)
            return key, {}

        image_data = {}

        aspect_ratio = height / width
        if height < min_resolution or width < min_resolution:
            return key, {}
        if aspect_ratio < min_aspect_ratio or aspect_ratio > 1 / min_aspect_ratio:
            return key, {}

        if multi_resolution:
            assert image_transform is not None

            images, patch_pos = process_anyres_image(image, image_transform,
                                                     grid_pinpoints,
                                                     base_resolution)
            image_data.update({
                'images':
                images,
                'images_patch_length':
                torch.tensor([images.shape[0]], dtype=torch.long),
                'patch_position':
                patch_pos,
                'image_size':
                torch.tensor([image.size], dtype=torch.long)
            })

        else:
            if image_transform is not None:
                clip_image_tensor = image_transform(image)
                image_data['images'] = clip_image_tensor
            else:
                image_data['images'] = image

        return key, image_data
    elif key.endswith(".json"):
        try:
            metadata_str = value.read().decode('utf-8')
            if use_caption_in_metadata and assure_text:
                metadata = json.loads(metadata_str)
                caption = metadata[caption_key_in_metadata]

                if add_gen_prompt:
                    num_ids = random.randint(0, len(gen_prompt_all) - 1)
                    gen_prompt = gen_prompt_all[num_ids]
                    gen_prompt_response = gen_prompt_response_all[num_ids]
                else:
                    gen_prompt = None
                    gen_prompt_response = None

                if gen_prompt is not None:
                    caption = gen_prompt + ' ' + caption.lstrip(' ')

                if instruction_prompt is not None:
                    caption = instruction_prompt.format_map(
                        {'instruction': caption})

                response = ''
                if gen_prompt_response is not None:
                    response = gen_prompt_response

                if tokenizer is None or multi_resolution:
                    return key, {'text': caption, 'response': response}
                else:

                    results = encode_caption_input_ids(
                        caption=caption,
                        tokenizer=tokenizer,
                        img_first_ratio=img_first_ratio,
                        max_length=max_length,
                        num_img_in_tokens=num_img_in_tokens,
                        num_img_out_tokens=num_img_out_tokens)

                    return key, {
                        'input_ids': results['input_ids'],
                        'attention_mask': results['attention_mask'],
                        'labels': results['labels'],
                        'ids_gen_mask': results['ids_gen_mask'],
                        'ids_cmp_mask': results['ids_cmp_mask'],
                        'embeds_gen_mask': results['embeds_gen_mask'],
                        'embeds_cmp_mask': results['embeds_cmp_mask'],
                        'text': caption
                    }
            else:
                return key, {'metadata': metadata_str}
        except Exception as e:
            print('Error while load metadata or encode caption: ', e)
            return key, {}
    else:
        return key, {}


def build_caption_datapipes_with_pixels(data_dir,
                                        tokenizer=None,
                                        max_length=77,
                                        batch_size=None,
                                        similarity_thr=0.2,
                                        min_resolution=180,
                                        image_transform=None,
                                        min_aspect_ratio=0.666,
                                        use_caption_in_metadata=False,
                                        instruction_prompt=None,
                                        turn_sep='\n',
                                        add_gen_prompt=False,
                                        caption_key_in_metadata='top_caption',
                                        img_first_ratio=0.5,
                                        num_img_in_tokens=64,
                                        num_img_out_tokens=64,
                                        assure_text=True,
                                        cycle_count=None,
                                        multi_resolution=False,
                                        resolution_grids=None,
                                        base_resolution=224,
                                        dataset_name=None):
    """
    datapipe of caption dataset (such as CC3M, LAION...) with webdataset format
    """
    grid_pinpoints = []
    if multi_resolution:
        resolution_grids = list(resolution_grids)

        for scale in resolution_grids:
            s1, s2 = scale.split('x')
            grid_pinpoints.append(
                [int(s1) * base_resolution,
                 int(s2) * base_resolution])

    decode_partial = functools.partial(
        decode_image_text_pair,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=max_length,
        use_caption_in_metadata=use_caption_in_metadata,
        instruction_prompt=instruction_prompt,
        add_gen_prompt=add_gen_prompt,
        caption_key_in_metadata=caption_key_in_metadata,
        min_resolution=min_resolution,
        min_aspect_ratio=min_aspect_ratio,
        img_first_ratio=img_first_ratio,
        num_img_in_tokens=num_img_in_tokens,
        num_img_out_tokens=num_img_out_tokens,
        assure_text=assure_text,
        multi_resolution=multi_resolution,
        resolution_grids=resolution_grids,
        base_resolution=base_resolution,
        grid_pinpoints=grid_pinpoints)

    filter_partial = functools.partial(filter_data_with_similarity,
                                       similarity_thr=similarity_thr,
                                       assure_text=assure_text)

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

    if multi_resolution:
        tokenize_func = functools.partial(
            tokenize_text,
            tokenizer=tokenizer,
            turn_sep=turn_sep,
            img_first_ratio=img_first_ratio,
            max_length=max_length,
            num_img_in_tokens=num_img_in_tokens,
            num_img_out_tokens=num_img_out_tokens)

        datapipe = datapipe.map(tokenize_func)

    datapipe = datapipe.filter(filter_partial)
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
            datapipe = datapipe.collate()

    return datapipe
