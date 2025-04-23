from abc import abstractmethod

import pytorch_lightning as pl
from torch import FloatTensor

from embedding_studio.embeddings.features.fine_tuning_features import (
    FineTuningFeatures,
)


class RankingLossInterface(pl.LightningModule):
    """
    Abstract interface for ranking loss functions used in embedding fine-tuning.

    This interface extends PyTorch Lightning's LightningModule and defines
    the contract for ranking loss implementations. Subclasses must implement
    the __call__ method to calculate loss from FineTuningFeatures.
    """

    @abstractmethod
    def __call__(self, features: FineTuningFeatures) -> FloatTensor:
        """
        Calculate the ranking loss based on provided features.

        :param features: Fine-tuning features containing positive and negative examples
                         along with their confidences and targets
        :return: Calculated loss as a float tensor

        Example implementation:
        def __call__(self, features: FineTuningFeatures) -> FloatTensor:
            # Calculate difference between positive and negative examples
            diff = features.positive_ranks - features.negative_ranks

            # Apply loss function to the difference
            loss = some_loss_function(diff, features.target)

            return loss
        """
        raise NotImplemented()
