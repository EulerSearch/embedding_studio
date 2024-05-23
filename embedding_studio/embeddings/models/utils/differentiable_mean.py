import torch

from embedding_studio.embeddings.models.utils.soft_indicator import (
    soft_indicator,
)


def differentiable_mean_small_values(
    x: torch.FloatTensor, threshold: float = 0.01, steepness: int = 100
) -> torch.FloatTensor:
    """Differentiable version of torch.mean(x[x<threshold]).

    :param x: tensor values
    :param threshold: threshold value, near which indicator value is ~ 0.5
    :param steepness: sharpness parameter
    :return: mean values, that close to the real mean, but can be differentiated
    """
    weights = soft_indicator(x, threshold, steepness)
    weighted_abs_diff = torch.abs(x) * weights
    sum_weighted_abs_diff = torch.sum(weighted_abs_diff)
    sum_weights = torch.sum(weights)
    # Avoid division by zero
    differentiable_mean = sum_weighted_abs_diff / (sum_weights + 1e-10)
    return differentiable_mean
