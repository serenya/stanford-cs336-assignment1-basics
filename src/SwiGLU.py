import torch
import torch.nn as nn

class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int
    ):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_ff, d_model)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)

    def positionwise_feedforward(
        self,
        in_features: torch.Tensor
    ) -> torch.Tensor:
        w1_in_features = in_features @ self.w1.weight.T
        silu = w1_in_features * torch.sigmoid(w1_in_features)
        w3_in_features = in_features @ self.w3.weight.T
        return (silu * w3_in_features) @ self.w2.weight.T

    