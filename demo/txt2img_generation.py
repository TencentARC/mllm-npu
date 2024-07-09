import hydra
import argparse
import torch
import os
import json
import pyrootutils
from omegaconf import OmegaConf
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, EulerDiscreteScheduler

import torch_npu
from torch_npu.contrib import transfer_to_npu


def main(args):
    BOI_TOKEN = '<img>'
    EOI_TOKEN = '</img>'
    IMG_TOKEN = '<img_{:05d}>'

    device = 'cuda:0'
    dtype = torch.float16
    dtype_str = 'fp16'
    num_img_in_tokens = 64
    num_img_out_tokens = 64

    instruction_prompt = '{caption}<img>'

    configs = json.load(open(args.config_file, "r"))
    model_cfg = OmegaConf.load(configs["mllm_model"]).mllm

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

    prompt = instruction_prompt.format_map({'caption': args.input_text})
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokenizer.bos_token_id] + prompt_ids).to(device, dtype=torch.long).unsqueeze(0)
    output = mllm_model.generate(tokenizer=tokenizer, input_ids=input_ids, num_img_gen_tokens=num_img_out_tokens)

    vision_encoder = mllm_model.vision_encoder

    del mllm_model
    del llm_model
    torch.cuda.empty_cache()

    diffusion_model_path = configs["diffusion_model"]

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
    print('init vae')
    vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device, dtype=dtype)
    print('init unet')
    unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device, dtype=dtype)

    adapter_cfg_path = configs["adapter"]
    adapter_cfg = OmegaConf.load(adapter_cfg_path)
    adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device, dtype=dtype).eval()

    discrete_model_cfg_path = configs["discrete_model"]
    discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
    discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device).eval()
    print('Init adapter done')

    image_transform_cfg = model_cfg.processor
    image_transform_cfg["_target_"] = "mllm_npu.data.processor.image_processing_clip.CLIPImageProcessor"
    image_transform = hydra.utils.instantiate(image_transform_cfg)

    adapter.init_pipe(
        vae=vae,
        scheduler=noise_scheduler,
        visual_encoder=vision_encoder,
        image_transform=image_transform,
        discrete_model=discrete_model,
        dtype=dtype,
        device=device
    )

    print('Init adapter pipe done')

    images = adapter.generate(image_embeds=output['img_gen_feat'], num_inference_steps=50)
    save_path = './demo/test.png'
    images[0].save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='txt2img generation demo')

    parser.add_argument('--input_text', type=str, default='A car was parked next to a wooden house.')
    parser.add_argument('--config_file', type=str, default='./mllm_npu/configs/workers/seedx_workers.json')

    args = parser.parse_args()

    main(args)