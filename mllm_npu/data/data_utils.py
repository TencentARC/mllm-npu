import json
import io
import base64

import torch
from PIL import Image


def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img_buffer = io.BytesIO(img_data)
    img = Image.open(img_buffer).convert('RGB')
    return img


def custom_collate(batch, dataset_name=None):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [
            batch[i][key] for i in range(len(batch))
            if batch[i][key] is not None
        ]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    results['dataset_name'] = dataset_name

    return results


def mmc4_collate(batch):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [
            batch[i][key] for i in range(len(batch))
            if batch[i][key] is not None
        ]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            if key in ['embeds_gen_mask', 'embeds_cmp_mask', 'images']:
                results[key] = torch.cat(cur, dim=0)
            else:
                results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    return results


def llava_collate(batch):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [
            batch[i][key] for i in range(len(batch))
            if batch[i][key] is not None
        ]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    return results


def filter_data_with_image_ids(item):
    if ('images' not in item):
        return False
    elif 'input_ids' not in item:
        return False
    else:
        return True


def filter_data_with_similarity(item, similarity_thr=0.2, assure_text=True):
    """
    get the image text similarity in data
    """
    if ('images' not in item):
        return False
    elif (not item.get('filter_flag', True)):
        return False
    elif assure_text and ('text' not in item):
        return False
    else:
        metadata = json.loads(item['metadata'])

        if 'all_similarities' in metadata:
            similarity = max(metadata['all_similarities'])
        elif 'similarity' in metadata:
            similarity = metadata['similarity']
        elif 'score' in metadata:
            similarity = metadata['score']
        elif 'SCORE' in metadata:
            similarity = metadata['SCORE']
        else:
            similarity = None

        if similarity is not None:
            if similarity < similarity_thr:
                return False

        return True


def unwarp_data(item):
    unwarpped = {}
    for key, value in item.items():
        if isinstance(value, dict):
            unwarpped.update(value)
        elif value is not None:
            unwarpped[key] = value
    if 'metadata' not in unwarpped:
        unwarpped['metadata'] = '{}'

    return unwarpped


def select(sample):
    ret = {
        'input_ids': sample['input_ids'],
        'attention_mask': sample['attention_mask'],
        'labels': sample['labels'],
        'ids_gen_mask': sample['ids_gen_mask'],
        'ids_cmp_mask': sample['ids_cmp_mask'],
        'embeds_gen_mask': sample['embeds_gen_mask'],
        'embeds_cmp_mask': sample['embeds_cmp_mask'],
        'images': sample['images'],
    }
    for k in ['images_patch_length', 'patch_position', 'image_size']:
        if k in sample:
            ret[k] = sample[k]

    return ret
