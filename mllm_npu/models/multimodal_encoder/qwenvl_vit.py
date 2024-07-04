# Copyright (c) Alibaba Cloud.
import os
import math
import requests
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional, List

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mllm_npu.utils import load_zero3_checkpoint
from mllm_npu.models.multimodal_projector.attention_resampler import AttentionResampler
from mllm_npu.models.multimodal_projector.attention_resampler import get_abs_pos


class VisualAttention(nn.Module):
    """self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, embed_dim, num_heads, bias=True, kdim=None, vdim=None):
        super(VisualAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

        # Per attention head and per partition values.
        assert embed_dim % num_heads == 0
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        # Strided linear layer.
        assert self._qkv_same_embed_dim, 'Only Support SelfAttention Currently'
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(self, query, key, value, attn_mask=None):
        # query/key/value: [sq, b, h]
        sq, b, _ = query.size()

        #  query is key, 'Only Support Self-Attention Currently'
        sk = sq
        mixed_x_layer = self.in_proj(query)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = mixed_x_layer.split(
            self.hidden_size_per_attention_head, dim=-1)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            sq, b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(
            sk, b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        q_scaled = query_layer / self.norm_factor
        if attn_mask is not None:
            attention_probs = torch.baddbmm(attn_mask, q_scaled,
                                            key_layer.transpose(-2, -1))
        else:
            attention_probs = torch.bmm(q_scaled, key_layer.transpose(-2, -1))
        attention_probs = attention_probs.softmax(dim=-1)

        value_layer = value_layer.view(
            sk, b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(
            b, self.num_attention_heads_per_partition, sq,
            self.hidden_size_per_attention_head)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.out_proj(context_layer)

        return output


class VisualAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, mlp_width)),
                         ("gelu", act_layer()),
                         ("c_proj", nn.Linear(mlp_width, d_model))]))

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(
            self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(
            self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.attention(
            q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([
            VisualAttentionBlock(width,
                                 heads,
                                 mlp_ratio,
                                 act_layer=act_layer,
                                 norm_layer=norm_layer) for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def get_cast_device(self) -> torch.device:
        return self.resblocks[0].mlp.c_fc.weight.device

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):

        gradient_checkpointing = True
        for r in self.resblocks:
            if gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(r, x, None, None,
                                                      attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformerWithAttnPool(nn.Module):

    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float,
                 n_queries: int = 256,
                 output_dim: int = 512,
                 patch_pos: bool = False,
                 **kwargs):
        super().__init__()
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height,
                          image_width // patch_width)
        self.output_dim = output_dim

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale *
                                                 torch.randn(256, width))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.ln_pre = norm_layer(width)
        self.transformer = TransformerBlock(
            width,
            layers,
            heads,
            mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.attn_pool = AttentionResampler(
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=output_dim,
            num_heads=output_dim // 128,
            kv_dim=width,
            norm_layer=norm_layer,
        )

        self.patch_pos = patch_pos
        if patch_pos:
            # 4*dim for the 4 corners of the image
            self.patch_pos_embed = nn.Parameter(
                (output_dim**-0.5) * torch.randn(4, output_dim))

        self.ln_post = norm_layer(output_dim)
        self.proj = nn.Parameter(
            (output_dim**-0.5) * torch.randn(output_dim, output_dim))

    def forward(self,
                x: torch.Tensor,
                patch_positions: Optional[torch.Tensor] = None):
        x = x.to(
            dtype=self.transformer.get_cast_dtype(),
            device=self.transformer.get_cast_device(),
        )
        # to patches
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = x + get_abs_pos(self.positional_embedding, x.size(1))

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.attn_pool(x)
        if self.patch_pos:
            patch_positions = patch_positions.to(
                dtype=self.transformer.get_cast_dtype(),
                device=self.transformer.get_cast_device(),
            )
            rel_posembed = torch.mm(
                torch.cat([patch_positions, 1 - patch_positions], dim=-1) / 2,
                self.patch_pos_embed).unsqueeze(1)
            x = x + rel_posembed
        x = self.ln_post(x)
        x = x @ self.proj

        return x

    def encode(self, image_paths: List[str]):
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith(
                    "https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = torch.stack(images, dim=0)
        return self(images)

    @classmethod
    def from_pretrained(cls, pretrained_model_path=None, **kawrgs):
        if os.environ.get('DEBUG_FLAG', 'False') == 'True':
            print(
                'DEBUG_FLAG is set to True, return a random initialized model')
            kawrgs.update({
                "heads": 4,
                "image_size": 448,
                "layers": 1,
                "mlp_ratio": 1,
                "output_dim": 768,  # llama input dim
                "patch_size": 14,
                "width": 768,
            })
            return cls(**kawrgs)
        model = cls(**kawrgs)

        if pretrained_model_path is not None:
            # print('Load ckpt of qwen visual encoder')
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            load_zero3_checkpoint(model, ckpt)

        return model


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float,
                 n_queries: int = 256,
                 output_dim: int = 512,
                 **kwargs):
        super().__init__()
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height,
                          image_width // patch_width)
        self.output_dim = output_dim

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale *
                                                 torch.randn(256, width))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.ln_pre = norm_layer(width)
        self.transformer = TransformerBlock(
            width,
            layers,
            heads,
            mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor):
        x = x.to(
            dtype=self.transformer.get_cast_dtype(),
            device=self.transformer.get_cast_device(),
        )
        # to patches
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = x + get_abs_pos(self.positional_embedding, x.size(1))

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x

    def encode(self, image_paths: List[str]):
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith(
                    "https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = torch.stack(images, dim=0)
        return self(images)
