from typing import Literal

import torch
import torch.nn.functional as F


def differentiable_extreme(
    x: torch.Tensor, beta: float = 1e5, mode: Literal["max", "min"] = "max"
):
    """
    Approximates the max or min function in a differentiable manner using softmax.

    Parameters:
    - x: Tensor, input tensor.
    - beta: float, scaling parameter for the softmax function. A larger beta makes
            the softmax output closer to the actual max or min function.
    - mode: str, either 'max' or 'min'. Determines whether to approximate the max or min function.

    Returns:
    - Tensor, the result of the differentiable max or min operation.
    """
    if mode == "max":
        softmax = F.softmax(x * beta, dim=-1)
    elif mode == "min":
        softmax = F.softmax(-x * beta, dim=-1)
    else:
        raise ValueError("Mode must be either 'max' or 'min'")

    # Weighted sum to approximate max or min
    diff_extreme = torch.sum(softmax * x, dim=-1)

    return diff_extreme
