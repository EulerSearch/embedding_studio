from typing import Optional

from torch import FloatTensor

from embedding_studio.embeddings.losses.prob_margin_ranking_loss import (
    ProbMarginRankingLoss,
)


class CosineProbMarginRankingLoss(ProbMarginRankingLoss):
    def __init__(self, base_margin: Optional[float] = 1.0):
        """Embeddings Fine-tuning Loss (modification of MarginRankingLoss)
        Use sigmoid instead of ReLU + results confidences to ignore noises and mistakes.
        Adapt to cosine similarity / distance

        :param base_margin: margin ranking loss margin (default: 1.0)
        """
        super(CosineProbMarginRankingLoss, self).__init__(
            base_margin=base_margin
        )

    def __adjust(self, adjusted_diff: FloatTensor) -> FloatTensor:
        # The way any wrong difference more than 0.01 is worth to be penaltized
        # Sigmoid with this kind of input return prob > 0.1, for difference between
        # pos and more than 0.001. That's our expected behaviour.
        # TODO: implement calculation of magic numbers
        return -400 * adjusted_diff + 6
