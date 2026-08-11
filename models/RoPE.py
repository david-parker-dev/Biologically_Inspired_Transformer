import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_vector_values(self, x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x, positions):

        # m * theta_i, shape (seq, head_dim/2)
        angles = torch.outer(positions.float(), self.inv_freq)
        angles = torch.stack([angles, angles], dim=-1).flatten(-2)                    

        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        result = (x * cos) + (self._rotate_vector_values(x) * sin)

        return result