from abc import abstractmethod

import pytorch_lightning as pl
from torch import FloatTensor

from embedding_studio.embeddings.features.fine_tuning_features import (
    FineTuningFeatures,
)


class RankingLossInterface(pl.LightningModule):
    @abstractmethod
    def __call__(self, features: FineTuningFeatures) -> FloatTensor:
        raise NotImplemented()
