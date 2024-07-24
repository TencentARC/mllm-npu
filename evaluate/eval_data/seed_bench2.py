import os
import os.path as osp
import sys
import json
import copy
import hydra
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from mllm_npu.data.utils import process_anyres_image
from mllm_npu.constant import (BOI_TOKEN, EOI_TOKEN, BOP_TOKEN, EOP_TOKEN,
                               IMG_TOKEN, dynamic_padding)
import numpy as np
import random

import torch_npu
from torch_npu.contrib import transfer_to_npu

# root directory of cc3m
cc3m_dir = "cc3m-image"
# root directory of seed bench v2
seed_bench_v2_dir = "SEED-Bench-2-image"

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MLLM_Tester(nn.Module):
    def __init__(self, mllm_model, tokenizer, image_transform):
        super().__init__()

        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.dtype_str = 'bf16'

        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.mllm_model = mllm_model

        self.num_img_in_tokens = 64
        self.num_img_out_tokens = 64
        self.max_length = 1024
        self.base_resolution = 448
        self.multi_resolution = True
        self.resolution_grids = ['1x1', '1x2', '1x3', '2x1', '3x1', '1x4', '4x1', '2x2']

        self.grid_pinpoints = []
        for scale in self.resolution_grids:
            s1, s2 = scale.split('x')
            self.grid_pinpoints.append(
                [int(s1) * self.base_resolution,
                 int(s2) * self.base_resolution])

    def forward(self, x):
        data_path, question, choices = x['data_path'], x['question'], x['choices']

        all_losses = []
        with torch.no_grad():
            for cand in choices:
                input_text = "Question: {}\nAnswer: {}".format(question, cand)
                if type(data_path) == list:
                    num_imgs = len(data_path) - len(input_text.split("<img>")) + 1
                    for _ in range(num_imgs):
                        input_text = "<img>" + input_text
                elif type(data_path) == str:
                    input_text = "<img>\n" + input_text
                    data_path = [data_path]

                input_text = input_text.replace("<img>", "None|||")
                text_list = input_text.split("|||")
                image_list = []

                for k in range(len(text_list)):
                    if text_list[k] == "None":
                        text_list[k] = None
                        image_list.append(Image.open(data_path[len(image_list)]).convert('RGB'))
                    else:
                        image_list.append(None)

                images = []
                input_ids = [self.tokenizer.bos_token_id]
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

                for image_pil, text in zip(image_list, text_list):
                    if image_pil is not None:
                        if self.multi_resolution:
                            img_size = image_pil.size
                            image, patch_pos = process_anyres_image(
                                image_pil, self.image_transform, self.grid_pinpoints,
                                self.base_resolution)

                            if cur_seq_length + (self.num_img_in_tokens + 2) * len(
                                    patch_pos) >= self.max_length:
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
                                    for item in range(self.num_img_in_tokens)
                                ]) + EOP_TOKEN
                            image_tokens += BOI_TOKEN + ''.join([
                                IMG_TOKEN.format(int(item))
                                for item in range(self.num_img_in_tokens)
                            ]) + EOI_TOKEN

                            image_ids = self.tokenizer.encode(
                                image_tokens, add_special_tokens=False)
                            image_labels = [-100] * len(image_ids)

                            for i in range(len(patch_pos)):
                                ids_cmp_mask.extend([False] + [True] *
                                                    self.num_img_in_tokens +
                                                    [False])
                                ids_gen_mask.extend([False] + [False] *
                                                    self.num_img_in_tokens +
                                                    [False])

                        if self.image_transform is not None and not self.multi_resolution:
                            image = self.image_transform(image)

                        images.append(image)

                        input_ids.extend(image_ids)
                        labels.extend(image_labels)
                        cur_seq_length += len(image_ids)
                        input_text += image_tokens

                    else:
                        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
                        if cur_seq_length + len(text_ids) >= self.max_length:
                            break
                        input_ids.extend(text_ids)
                        ids_cmp_mask.extend([False] * len(text_ids))
                        ids_gen_mask.extend([False] * len(text_ids))
                        labels.extend(text_ids)
                        cur_seq_length += len(text_ids)
                        input_text += text

                input_ids.append(self.tokenizer.eos_token_id)
                labels.append(self.tokenizer.eos_token_id)
                ids_cmp_mask.append(False)
                ids_gen_mask.append(False)
                attention_mask = [1] * len(input_ids)
                if len(input_ids) >= self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    labels = labels[:self.max_length]
                    ids_gen_mask = ids_gen_mask[:self.max_length]
                    ids_cmp_mask = ids_cmp_mask[:self.max_length]

                elif not dynamic_padding:
                    padding_length = self.max_length - len(input_ids)
                    input_ids = input_ids + [self.tokenizer.pad_token_id
                                             ] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                    labels = labels + [-100] * padding_length
                    ids_gen_mask = ids_gen_mask + [False] * padding_length
                    ids_cmp_mask = ids_cmp_mask + [False] * padding_length

                if self.multi_resolution:
                    if len(images) > 0:
                        images = torch.cat(images)

                assert len(images) == len(embeds_cmp_mask) and len(images) == len(
                    embeds_gen_mask)
                if self.image_transform is not None and len(
                        images) > 0 and not self.multi_resolution:
                    images = torch.stack(images)

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

                input_ids = input_ids.unsqueeze(0).to(self.device)
                images = images.to(self.device)
                attention_mask = attention_mask.unsqueeze(0).to(self.device)
                labels = labels.unsqueeze(0).to(self.device)
                patch_positions = torch.cat(patch_position, dim=0).to(self.device)
                embeds_gen_mask = embeds_gen_mask.unsqueeze(0).to(self.device)
                embeds_cmp_mask = embeds_cmp_mask.to(self.device)
                ids_gen_mask = ids_gen_mask.unsqueeze(0).to(self.device)
                ids_cmp_mask = ids_cmp_mask.unsqueeze(0).to(self.device)

                output = self.mllm_model(
                    input_ids=input_ids,
                    images=images,
                    attention_mask=attention_mask,
                    labels=labels,
                    patch_positions=patch_positions,
                    embeds_gen_mask=embeds_gen_mask,
                    embeds_cmp_mask=embeds_cmp_mask,
                    ids_gen_mask=ids_gen_mask,
                    ids_cmp_mask=ids_cmp_mask
                )

                all_losses.append(output['lm_loss'].cpu())

        return all_losses


def build(mllm_model, tokenizer, image_transform):
    return MLLM_Tester(mllm_model, tokenizer, image_transform)


def filter_questions(data, level='L2', subpart='all', version='v2'):
    if level == "L1":
        valid_level_data = ['L1']
    elif level == "L2":
        valid_level_data = ['L1', 'L2']
    elif level == "L3":
        valid_level_data = ['L1', 'L2', 'L3']
    else:
        raise ValueError(f"Invalid level: {level}")
    data = [q for q in data if q["level"] in valid_level_data]

    if subpart in ['Single-Image & Text Comprehension', 'Multiple-Images & Text Comprehension',
                   'Video & Text Comprehension', 'Interleaved Image & Text Comprehension', 'Image Generation',
                   'Image & Text Generation']:
        valid_subgroup_data = subpart
    elif subpart == 'all':
        valid_subgroup_data = ['Single-Image & Text Comprehension', 'Multiple-Images & Text Comprehension',
                               'Video & Text Comprehension', 'Interleaved Image & Text Comprehension',
                               'Image Generation', 'Image & Text Generation']
    else:
        raise ValueError(f"Invalid subpart: {subpart}")
    data = [q for q in data if q["subpart"] in valid_subgroup_data]

    if version == 'v1':
        valid_version_data = ['v1']
    elif version == 'v2':
        valid_version_data = ['v1', 'v2']
    else:
        raise ValueError(f"Invalid version: {version}")
    data = [q for q in data if q["version"] in valid_version_data]

    return data


def run_inference(model, qa_anno, data_root):
    total_qa_num = len(qa_anno)
    answer_list = []
    output_f = open("results.json", "a")
    step = 0
    for qa_item in tqdm(qa_anno):
        data_info = {
            'question': qa_item['question'],
            'choices': [qa_item['choice_a'], qa_item['choice_b'], qa_item['choice_c'], qa_item['choice_d']],
        }

        if qa_item["data_source"] == 'cc3m':
            image_dir = os.path.join(data_root, cc3m_dir)
        elif qa_item["data_source"] == 'SEED-Bench v2':
            image_dir = os.path.join(data_root, seed_bench_v2_dir)
        else:
            raise ValueError("The data type is not valid.")
        if type(qa_item['data_id']) is list:
            data_path = [os.path.join(image_dir, path) for path in qa_item['data_id']]
        else:
            data_path = os.path.join(image_dir, qa_item['data_id'])
        data_info['data_path'] = data_path

        with torch.no_grad():
            losses = model(data_info)
        class_ranks = np.argsort(losses)
        pred_id = ['A', 'B', 'C', 'D'][class_ranks[0]]
        gt = qa_item['answer']
        answer_record = {
            'question_id': qa_item['question_id'],
            'prediction': pred_id
        }
        answer_list.append(answer_record)
        output_f.write(json.dumps(answer_record) + "\n")
        step += 1


def seed_bench2_eval(model, tokenizer, image_transform, data_root, device):
    qa_anno = json.load(open(os.path.join(data_root, "SEED-Bench_v2_level1_2_3.json"), 'rb'))
    if 'questions' in qa_anno.keys():
        qa_anno = qa_anno['questions']
    qa_anno = filter_questions(qa_anno)

    model = build(model, tokenizer, image_transform)
    run_inference(model, qa_anno, data_root)