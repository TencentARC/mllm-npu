from typing import Optional

import torch
import torch.nn as nn
from transformers.models.siglip.modeling_siglip import SiglipVisionModel


class SigLIPVisionEncoder(nn.Module):
    """
      "vision_config": {
        "hidden_size": 1152,
        "image_size": 384,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14
        }
    """

    def __init__(self,
                 vision_model,
                 hidden_dim: int,
                 output_dim: int,
                 patch_pos: bool = False,
                 **kwargs) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.patch_pos = patch_pos
        self.vision_model = vision_model

    def forward(self,
                x: torch.Tensor,
                patch_positions: Optional[torch.Tensor] = None):
        x = x.to(self.vision_model.vision_model.post_layernorm.weight)
        x = self.vision_model(x)
        # output 27x27 tokens, BxLxD (B, 729,1152)
        x = x['last_hidden_state']
        return x

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, **kwargs):
        vision_model = SiglipVisionModel.from_pretrained(
            pretrained_model_name_or_path)
        kwargs.update({"hidden_dim": 1152, "output_dim": 4096})
        print(f"loading SigLIP from: {pretrained_model_name_or_path}")
        model = cls(vision_model, **kwargs)
        return model
