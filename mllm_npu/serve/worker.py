import argparse
import asyncio
import uuid
import json
import requests
import threading
import time
import hydra
import torch
import os
import re
import cv2
import uuid
import uvicorn
import queue
import numpy as np
import io
import base64

from PIL import Image
from omegaconf import OmegaConf
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from functools import partial
from threading import Thread
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, EulerDiscreteScheduler
from mllm_npu.serve.serve_utils import build_logger, server_error_msg, pretty_print_semaphore
from mllm_npu.data.utils import process_anyres_image

import torch_npu
from torch_npu.contrib import transfer_to_npu


WORKER_HEART_BEAT_INTERVAL = 15

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register, device, config,
                 limit_model_concurrency):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.device = device

        logger.info(f"Loading the model seedx on worker {worker_id} ...")
        self.config = json.load(open(config, "r"))
        logger.info(f"config: {self.config}")

        model_cfg = OmegaConf.load(self.config["mllm_model"]).mllm

        logger.info(f"model_cfg: {model_cfg}")

        tokenizer_cfg = model_cfg.tokenizer
        self.tokenizer = hydra.utils.instantiate(tokenizer_cfg, add_prefix_space=False)

        image_transform_cfg = model_cfg.processor
        image_transform_cfg["_target_"] = "mllm_npu.data.processor.image_processing_clip.CLIPImageProcessor"
        self.image_transform = hydra.utils.instantiate(image_transform_cfg)

        self.dtype = torch.float16
        self.dtype_str = 'fp16'

        language_model_cfg = model_cfg.language_model
        self.llm = hydra.utils.instantiate(language_model_cfg, torch_dtype=self.dtype_str)
        logger.info(f"Init llm done")

        mllm_model_cfg = model_cfg.mllm_model
        self.mllm_model = hydra.utils.instantiate(mllm_model_cfg, language_model=self.llm)
        self.mllm_model.eval().to(self.device, dtype=self.dtype)
        logger.info(f"Init mllm model done")

        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(self.config["diffusion_model"],
                                                                      subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.config["diffusion_model"], subfolder="vae").to(self.device,
                                                                                                    dtype=self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(self.config["diffusion_model"], subfolder="unet").to(
            self.device, dtype=self.dtype)
        adapter_cfg = OmegaConf.load(self.config["adapter"])
        self.adapter = hydra.utils.instantiate(adapter_cfg, unet=self.unet).to(self.device, dtype=self.dtype).eval()
        discrete_model_cfg = OmegaConf.load(self.config["discrete_model"])
        self.discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(self.device).eval()
        logger.info(f"Init adapter model done")

        self.adapter.init_pipe(
            vae=self.vae,
            scheduler=self.noise_scheduler,
            visual_encoder=self.mllm_model.vision_encoder,
            image_transform=self.image_transform,
            discrete_model=self.discrete_model,
            dtype=self.dtype,
            device=self.device
        )
        logger.info(f"Init adapter pipe done")

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,), daemon=True)
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": ["seed-x"],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {['seedx']}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    @torch.inference_mode()
    def generate(self, params):
        tokenizer, image_transform, mllm_model = self.tokenizer, self.image_transform, self.mllm_model
        adapter = self.adapter

        BOI_TOKEN = '<img>'
        BOP_TOKEN = '<patch>'
        EOI_TOKEN = '</img>'
        EOP_TOKEN = '</patch>'
        IMG_TOKEN = '<img_{:05d}>'

        resolution_grids = ['1x1', '1x2', '1x3', '2x1', '3x1', '1x4', '4x1', '2x2']
        base_resolution = 448

        boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
        eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

        bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
        eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]

        num_img_in_tokens = 64
        num_img_out_tokens = 64

        instruction_prompt = '{caption}<img>'

        grid_pinpoints = []
        for scale in resolution_grids:
            s1, s2 = scale.split('x')
            grid_pinpoints.append([int(s1) * base_resolution, int(s2) * base_resolution])
        grid_pinpoints = grid_pinpoints

        if not params['image_gen']:
            decoded_data = base64.b64decode(params['image'])
            image_data = io.BytesIO(decoded_data)
            image = Image.open(image_data)
            image = image.convert('RGB')
            image_tensor, patch_pos_tensor = process_anyres_image(image, self.image_transform, grid_pinpoints,
                                                                  base_resolution)
            embeds_cmp_mask = torch.tensor([True] * image_tensor.shape[0]).to(self.device, dtype=torch.bool)

            patch_pos = [patch_pos_tensor]
            patch_position = torch.cat(patch_pos, dim=0)

            image_tensor = image_tensor.to(self.device, dtype=self.dtype)

            patch_length = image_tensor.shape[0]
            image_tokens = ''
            for _ in range(patch_length - 1):
                image_tokens += BOP_TOKEN + ''.join(
                    IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOP_TOKEN
            image_tokens += BOI_TOKEN + ''.join(
                IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN

            prompt = image_tokens + 'Question: {}\nAnswer:'.format(params['input_text'])

            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = [tokenizer.bos_token_id] + input_ids

            input_ids = torch.tensor(input_ids).to(self.device, dtype=torch.long)

            ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            boi_indices = torch.where(torch.logical_or(input_ids == boi_token_id, input_ids == bop_token_id))[
                0].tolist()
            eoi_indices = torch.where(torch.logical_or(input_ids == eoi_token_id, input_ids == eop_token_id))[
                0].tolist()

            for boi_idx, eoi_idx in zip(boi_indices, eoi_indices):
                ids_cmp_mask[boi_idx + 1:eoi_idx] = True

            input_ids = input_ids.unsqueeze(0)
            ids_cmp_mask = ids_cmp_mask.unsqueeze(0)

            def text_generate(
                    tokenizer, input_ids,
                    image_tensor,
                    embeds_cmp_mask,
                    patch_positions,
                    ids_cmp_mask,
                    max_new_tokens,
                    num_img_out_tokens,
                    queue
            ):
                with torch.no_grad():
                    output = mllm_model.generate(
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        image_embeds=image_tensor,
                        embeds_cmp_mask=embeds_cmp_mask,
                        patch_positions=patch_positions,
                        ids_cmp_mask=ids_cmp_mask,
                        max_new_tokens=max_new_tokens,
                        num_img_gen_tokens=num_img_out_tokens
                    )

                output_text = re.sub('<[^>]*>', '', output['text'])
                output_text = re.sub(r'\[(.*)\]', '', output_text)

                queue.put(output_text)

            with torch.no_grad():
                q = queue.Queue()
                thread = Thread(
                    target=text_generate,
                    kwargs=dict(
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        image_embeds=image_tensor,
                        embeds_cmp_mask=embeds_cmp_mask,
                        patch_positions=patch_position,
                        ids_cmp_mask=ids_cmp_mask,
                        max_new_tokens=512,
                        num_img_gen_tokens=num_img_out_tokens,
                        queue=q
                    )
                )
                thread.start()

            output = q.get()
            yield json.dumps({"text": output, "error_code": 0}).encode() + b"\0"
        elif params['image_gen']:
            prompt = instruction_prompt.format_map({'caption': params['input_text']})
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokenizer.bos_token_id] + prompt_ids).to(self.device, dtype=torch.long).unsqueeze(
                0)

            def image_generate(tokenizer, input_ids, num_img_out_tokens, mllm_model, adapter, queue):
                with torch.no_grad():
                    output = mllm_model.generate(
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        num_img_gen_tokens=num_img_out_tokens
                    )

                images = adapter.generate(image_embeds=output['img_gen_feat'], num_inference_steps=50)
                out_img = images[0]
                image_bytes = io.BytesIO()
                out_img.save(image_bytes, format='JPEG')
                base64_image = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

                queue.put(base64_image)

            with torch.no_grad():
                q = queue.Queue()
                thread = Thread(target=image_generate, kwargs=dict(
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    num_img_out_tokens=num_img_out_tokens,
                    mllm_model=mllm_model,
                    adapter=adapter,
                    queue=q
                ))
                thread.start()

            output = q.get()
            yield json.dumps({"text": "generate successed.", "image": output, "error_code": 0}).encode() + b"\0"

    def generate_gate(self, params):
        try:
            for x in self.generate(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate")
async def generate(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
                        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
                        default="http://localhost:21001")
    parser.add_argument("--config", type=str, default="./mllm_npu/configs/workers/seedx_workers.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)

    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.device,
                         args.config,
                         args.limit_model_concurrency)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

