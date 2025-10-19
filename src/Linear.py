import torch
import torch.nn as nn
from torch import Tensor

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # Apply the linear transformation
        return x @ self.weight.T

# Example usage
if __name__ == "__main__":
    # Create an instance of the Linear class
    model = Linear(input_dim=3, output_dim=2)
    print(model)

    # Test with a random input tensor
    input_tensor = torch.randn(1, 3)  # Batch size of 1, input_dim of 3
    output = model(input_tensor)
    print("Input:", input_tensor)
    print("Output:", output)
