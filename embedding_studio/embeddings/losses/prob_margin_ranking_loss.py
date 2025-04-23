import torch
from torch import FloatTensor, Tensor

from embedding_studio.embeddings.features.fine_tuning_features import (
    FineTuningFeatures,
)
from embedding_studio.embeddings.losses.ranking_loss_interface import (
    RankingLossInterface,
)
from embedding_studio.embeddings.models.utils.differentiable_mean import (
    differentiable_mean_small_values,
)


class ProbMarginRankingLoss(RankingLossInterface):
    """
    Probabilistic Margin Ranking Loss for embedding fine-tuning.

    A modification of standard MarginRankingLoss that uses sigmoid instead of ReLU
    and incorporates result confidences to reduce the impact of noise and mistakes
    in the training data.
    """

    def __init__(
        self, base_margin: float = 1.0, do_fine_small_difference: bool = False
    ):
        """
        Initialize the ProbMarginRankingLoss.

        :param base_margin: The margin used in the ranking loss calculation, must be positive
        :param do_fine_small_difference: Flag to enable additional loss term for small differences
                                        between positive and negative ranks that are less than
                                        the base_margin, which helps prevent ignored examples
                                        from ruining embeddings
        """

        if not isinstance(base_margin, (int, float)) or base_margin <= 0:
            raise ValueError("base_margin must be a positive numeric value")

        super(ProbMarginRankingLoss, self).__init__()
        self.base_margin = base_margin
        self.do_fine_small_difference = do_fine_small_difference

    def set_margin(self, margin):
        """
        Set a new value for the base margin.

        :param margin: The new margin value to use
        """
        self.base_margin = margin

    def __adjust(self, adjusted_diff: FloatTensor) -> FloatTensor:
        """
        Adjust the difference for sigmoid calculation.

        :param adjusted_diff: The difference value to adjust
        :return: Adjusted difference value
        """
        return -1 * adjusted_diff

    def forward(self, features: FineTuningFeatures) -> Tensor:
        """
        Forward pass to calculate the probabilistic margin ranking loss.

        :param features: Features containing positive and negative ranks, confidences, and targets
        :return: Calculated loss value
        """
        # Calculate positive - negative pair confidence
        confidences: FloatTensor = torch.min(
            features.positive_confidences, features.negative_confidences
        )

        pairwise_diff: FloatTensor = (
            features.positive_ranks - features.negative_ranks
        )

        fine = 0.0
        if self.do_fine_small_difference:
            fine = differentiable_mean_small_values(
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

    def __call__(self, features: FineTuningFeatures) -> Tensor:
        """
        Calculate the probabilistic margin ranking loss.

        :param features: Fine-tuning features containing ranks, confidences, and targets
        :return: Calculated loss value
        """
        return self.forward(features)
