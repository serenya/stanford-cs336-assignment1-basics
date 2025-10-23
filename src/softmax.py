import torch

def softmax(
    in_features: torch.Tensor,
    dim: int
) -> torch.Tensor:
    subtracted_in_features = in_features - torch.max(in_features)
    exp_tensor = torch.exp(subtracted_in_features)
    return exp_tensor / exp_tensor.sum(dim, keepdim=True)