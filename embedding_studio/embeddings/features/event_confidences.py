import math
from typing import Union

import torch
from torch import FloatTensor, Tensor


@torch.no_grad()
def dummy_confidences(
    ranks: FloatTensor, events: Tensor
) -> Union[Tensor, FloatTensor]:
    """Confidence = 1.0

    :param ranks: list of ranks from search results
    :type ranks: FloatTensor
    :param events: list of 0 if it's not an event, 1 if it's an event
    :type events: FloatTensor
    :return: list of confidences
    :rtype: Union[Tensor, FloatTensor]
    """
    return torch.ones(len(events))


@torch.no_grad()
def calculate_confidences(
    ranks: FloatTensor, events: Tensor, window_size: int = 3
) -> Union[Tensor, FloatTensor]:
    """Calculate events (clicks) and not events (not clicks) confidences the way:
    * either event is among events of the same type
    * either event is among events of another type with different ranks

    :param ranks: list of ranks from search results
    :type ranks: FloatTensor
    :param events: list of 0 if it's not an event, 1 if it's an event
    :type events: FloatTensor
    :param window_size: context window to check confidence (default: 3)
                        like <i - window_size // 2; i; i + window_size // 2 + 1>
    :type window_size: int
    :return: list of confidences
    :rtype: Union[Tensor, FloatTensor]
    """
    num_results: int = len(events)
    confidence_scores: Tensor = torch.zeros(num_results)
    position_scores: Tensor = torch.zeros(num_results)

    for i in range(num_results):
        start: int = max(0, i - window_size // 2)
        end: int = min(num_results, i + window_size // 2 + 1)

        # Extract the ranks and clicks in the sliding window
        window_ranks: FloatTensor = ranks[start:end]
        window_clicks: FloatTensor = events[start:end]

        # Calculate the average rank and click proportion in the window
        avg_rank: Tensor = torch.mean(window_ranks)
        click_proportion: Tensor = torch.mean(
            window_clicks.type(torch.float32)
        )

        # Assessing the similarity of the current rank to the window's average rank
        rank_similarity: Tensor = torch.abs(ranks[i] - avg_rank) / avg_rank
        # TODO: move to config or provide customizable function
        position_scores[i] = math.exp(-3 * (i + 1) / num_results - 0.3) + 0.25

        if events[i] == 1:
            confidence_scores[i] = (
                (1 - rank_similarity) * click_proportion
            ) + (1 - click_proportion) * rank_similarity
        else:
            confidence_scores[i] = (1 - click_proportion) * (
                1 - rank_similarity
            ) + (click_proportion * rank_similarity)

    # Normalize confidence scores to be between 0 and 1
    confidence_scores: FloatTensor = confidence_scores * position_scores
    confidence_scores = (confidence_scores - confidence_scores.min()) / (
        confidence_scores.max() - confidence_scores.min()
    )

    return confidence_scores
