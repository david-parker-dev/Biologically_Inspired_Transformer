import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor

    def __init__(self, head_dim, max_positions, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base


        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Precompute angles for every position we'll ever see
        positions = torch.arange(max_positions).float()
        angles = torch.outer(positions, inv_freq)
        angles = torch.stack([angles, angles], dim=-1).flatten(-2)

        # Cache cos/sin tables: (max_positions, head_dim)
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def _rotate_vector_values(self, x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x, positions):
        cos = self.cos_cache[positions][None, None, :, :]
        sin = self.sin_cache[positions][None, None, :, :]

        result = (x * cos) + (self._rotate_vector_values(x) * sin)

        return result
