import hydra
import torch
import argparse
from omegaconf import OmegaConf

import torch_npu
from torch_npu.contrib import transfer_to_npu

from evaluate.eval_data.mmlu import mmlu_eval
from evaluate.eval_data.cmmlu import cmmlu_eval
from evaluate.eval_data.bbh import bbh_eval
from evaluate.eval_data.ceval import ceval_eval
from evaluate.eval_data.seed_bench2 import seed_bench2_eval
from evaluate.eval_data.mm_vet import mm_vet_eval
from evaluate.eval_data.mme import mme_eval


def main(args):
    device = 'cuda'
    dtype = torch.bfloat16
    dtype_str = 'bf16'

    model_cfg = OmegaConf.load(args.config_path).mllm

    image_transform_cfg = model_cfg.processor
    image_transform_cfg["_target_"] = "mllm_npu.data.processor.image_processing_clip.CLIPImageProcessor"
    image_transform = hydra.utils.instantiate(image_transform_cfg)

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

    if args.dataset_name == "mmlu":
        mmlu_eval(mllm_model, tokenizer, args.data_path, device)
    elif args.dataset_name == "cmmlu":
        cmmlu_eval(mllm_model, tokenizer, args.data_path, device)
    elif args.dataset_name == "bbh":
        bbh_eval(mllm_model, tokenizer, args.data_path, device)
    elif args.dataset_name == "ceval":
        ceval_eval(mllm_model, tokenizer, args.data_path, device)
    elif args.dataset_name == "seed_bench":
        seed_bench2_eval(mllm_model, tokenizer, image_transform, args.data_path, device)
    elif args.dataset_name == "mme":
        mme_eval(mllm_model, tokenizer, image_transform, args.data_path, device)
    elif args.dataset_name == "mm_vet":
        mm_vet_eval(mllm_model, tokenizer, image_transform, args.data_path, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate mllm on npu')

    parser.add_argument('--config_path', type=str, default='./mllm_npu/configs/models/seedx_llama2_13b_qwenvl_vitg.yaml')
    parser.add_argument('--dataset_name', type=str, default='mmlu')
    parser.add_argument('--data_path', type=str, default='./evaluate/eval_data/mmlu/')

    args = parser.parse_args()

    main(args)
