from typing import Optional

from torch import FloatTensor

from embedding_studio.embeddings.losses.prob_margin_ranking_loss import (
    ProbMarginRankingLoss,
)


class CosineProbMarginRankingLoss(ProbMarginRankingLoss):
    """
    Cosine-specific implementation of Probabilistic Margin Ranking Loss.

    This variant is adapted specifically for cosine similarity/distance metrics
    and applies a different adjustment function compared to the base class.
    """

    def __init__(self, base_margin: Optional[float] = 1.0):
        """
        Initialize the CosineProbMarginRankingLoss.

        :param base_margin: The margin used in the ranking loss calculation (default: 1.0)
        """
        super(CosineProbMarginRankingLoss, self).__init__(
            base_margin=base_margin
        )

    def __adjust(self, adjusted_diff: FloatTensor) -> FloatTensor:
        """
        Custom adjustment function optimized for cosine similarity.

        This function applies a specific transformation to ensure that differences
        above 0.01 are penalized. The resulting sigmoid will return probability > 0.1
        for differences between positives and negatives greater than 0.001.

        :param adjusted_diff: The difference value to adjust
        :return: Adjusted difference value
        """
        # The way any wrong difference more than 0.01 is worth to be penaltized
        # Sigmoid with this kind of input return prob > 0.1, for difference between
        # pos and more than 0.001. That's our expected behaviour.
        # TODO: implement calculation of magic numbers
        return -400 * adjusted_diff + 6
