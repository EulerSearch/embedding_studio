import torch
from torch import FloatTensor, Tensor

from embedding_studio.embeddings.features.session_features import (
    SessionFeatures,
)
from embedding_studio.embeddings.losses.differentiable_mean import (
    differentiable_mean,
)
from embedding_studio.embeddings.losses.ranking_loss_interface import (
    RankingLossInterface,
)


class ProbMarginRankingLoss(RankingLossInterface):
    def __init__(
        self, base_margin: float = 1.0, do_fine_small_difference: bool = False
    ):
        """Embeddings Fine-tuning Loss (modification of MarginRankingLoss)
        Use sigmoid instead of ReLU + results confidences to ignore noises and mistakes.

        :param base_margin: margin ranking loss margin (default: 1.0)
        :param do_fine_small_difference: margin ranking loss and triplet loss is quite greedy,
                and sometimes leads to the situation, when  |positive_ranks - negative_ranks| < base_margin,
                so these examples are ignored in the final loss, so we can get totally ruined embeddings.
                To avoid this situation we can recommend you to use the fine addition, which will fine embedding model
                for the small differences between positive_ranks and negative_ranks.
        """

        if not isinstance(base_margin, (int, float)) or base_margin <= 0:
            raise ValueError("base_margin must be a positive numeric value")

        super(ProbMarginRankingLoss, self).__init__()
        self.base_margin = base_margin
        self.do_fine_small_difference = do_fine_small_difference

    def set_margin(self, margin):
        self.base_margin = margin

    def __adjust(self, adjusted_diff: FloatTensor) -> FloatTensor:
        return -1 * adjusted_diff

    def forward(self, features: SessionFeatures) -> Tensor:
        # Calculate positive - negative pair confidence
        confidences: FloatTensor = torch.min(
            features.positive_confidences, features.negative_confidences
        )

        pairwise_diff: FloatTensor = (
            features.positive_ranks - features.negative_ranks
        )

        fine = 0.0
        if self.do_fine_small_difference:
            fine = differentiable_mean(
                pairwise_diff,
                self.base_margin,
                int(1 / self.base_margin * 100),
            )

        adjusted_diff: FloatTensor = (
            -features.target * pairwise_diff + self.base_margin
        )

        # Apply adjusted sigmoid function to the scaled and shifted adjusted_diff
        losses: FloatTensor = 1 / (1 + torch.exp(self.__adjust(adjusted_diff)))
        return torch.mean(losses * confidences) + fine

    def __call__(self, features: SessionFeatures) -> Tensor:
        return self.forward(features)
