import torch
import functools

import torchdata.datapipes as dp
from braceexpand import braceexpand

from mllm_npu.data.data_utils import (filter_data_with_image_ids, select,
                                      custom_collate)


def decode_text_pretrain_data_for_llm(item, tokenizer=None, max_length=512):
    key, value = item
    input_ids = []
    labels = []

    text = value.get('text', None)
    if text is None:
        return {}

    if text.strip(' ;,[]{}\'\".?:') == '':
        return {}

    if tokenizer is None:
        return {'text': text}

    tokenized = tokenizer(tokenizer.bos_token + text + tokenizer.eos_token,
                          max_length=max_length,
                          add_special_tokens=False,
                          truncation=True,
                          padding='max_length',
                          return_tensors='pt')

    input_ids = tokenized['input_ids'][0]
    attention_mask = tokenized['attention_mask'][0]
    labels = torch.clone(input_ids)
    labels[labels == tokenizer.pad_token_id] = -100

    ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    ids_gen_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    embeds_cmp_mask = None
    embeds_gen_mask = None

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        'images': None,
        'text': text,
    }


def build_text_pretrain_datapipes_for_llm(data_dir,
                                          tokenizer=None,
                                          image_transform=None,
                                          max_length=512,
                                          batch_size=None,
                                          cycle_count=None,
                                          dataset_name=None):
    """
    datapipe of pure text data
    """
    decode_partial = functools.partial(decode_text_pretrain_data_for_llm,
                                       tokenizer=tokenizer,
                                       max_length=max_length)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))

    datapipe = dp.iter.FileLister(root=data_dir,
                                  masks='*.jsonl',
                                  recursive=True)
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.shuffle(buffer_size=512)
    datapipe = datapipe.filter(filter_data_with_image_ids)
    datapipe = datapipe.map(select)

    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        collate_func = functools.partial(custom_collate,
                                         dataset_name=dataset_name)
        datapipe = datapipe.collate(collate_fn=collate_func)

    return datapipe
