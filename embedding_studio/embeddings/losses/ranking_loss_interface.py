from abc import abstractmethod

import pytorch_lightning as pl
from torch import FloatTensor

from embedding_studio.embeddings.features.session_features import (
    SessionFeatures,
)


class RankingLossInterface(pl.LightningModule):
    @abstractmethod
    def __call__(self, features: SessionFeatures) -> FloatTensor:
        raise NotImplemented()
