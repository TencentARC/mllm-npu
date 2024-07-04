import math
import torch.nn as nn


class SimplePooling(nn.Module):

    def __init__(self, grid_size, input_dim, output_dim) -> None:
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim)
        self.pooling = nn.AdaptiveAvgPool2d(grid_size)

    def forward(self, x):
        # BxLxD
        bz, l, d = x.shape
        s = int(math.sqrt(l))
        x = x.view(bz, s, s, d).permute(0, 3, 1, 2).contiguous()
        x = self.pooling(x)
        x = x.view(bz, d, -1).transpose(1, 2).contiguous()
        x = self.projector(x)
        return x
