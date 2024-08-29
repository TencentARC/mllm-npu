import math
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, image_embed_dim, llm_embed_dim) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(image_embed_dim * 4),
                                 nn.Linear(image_embed_dim * 4, llm_embed_dim),
                                 nn.GELU(),
                                 nn.Linear(llm_embed_dim, llm_embed_dim))
        self.embed_dim = llm_embed_dim

    def forward(self, x):
        # BxLxD
        return self.mlp(x)