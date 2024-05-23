import torch


def soft_indicator(
    x: torch.FloatTensor, threshold: float = 0.01, steepness: int = 100
):
    """Indicator that sensitive to the difference between x values and threshold.
    Works like  differentiable version of x < threshold.

    :param x: tensor values
    :param threshold: threshold value, near which indicator value is ~ 0.5
    :param steepness: sharpness parameter
    :return: weights from 0.0 to 1.0
    """
    # A steepness parameter controls the transition sharpness.
    return torch.sigmoid(steepness * (threshold - torch.abs(x)))
