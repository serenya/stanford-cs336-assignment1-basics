import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None
    ):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        # Generate position indices
        positions = torch.arange(self.max_seq_len).unsqueeze(1)  # Shape: (seq_len, 1)
        dim_indices = torch.arange(self.d_k // 2).unsqueeze(0)  # Shape: (1, d_k // 2)

        theta_positions = 1.0 / (self.theta ** (2 * dim_indices / self.d_k))

        # Compute rotation angles
        position_angles = positions * theta_positions

        # Create rotation matrix
        sin_angles = torch.sin(position_angles)
        cos_angles = torch.cos(position_angles)

        # Split query into even and odd dimensions
        x_even, x_odd = x[:, :, ::2], x[:, :, 1::2]

        # Apply rotation
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, ::2] = x_even * cos_angles - x_odd * sin_angles
        x_rotated[:, :, 1::2] = x_even * sin_angles + x_odd * cos_angles

        return x_rotated