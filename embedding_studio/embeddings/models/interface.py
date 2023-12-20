from abc import abstractmethod
from typing import Any, Iterator, List

import pytorch_lightning as pl
from torch import FloatTensor
from torch.nn import Parameter


class EmbeddingsModelInterface(pl.LightningModule):
    def __init__(self, same_query_and_items: bool = False):
        """In search we have two entities, which could be multi domain: query and search result (item).
        This is the interface we used in fine-tuning procedure.

        :param same_query_and_items: are query and items models acutally the same model (default: False)
        """
        super(EmbeddingsModelInterface, self).__init__()
        self.same_query_and_items = same_query_and_items

    @abstractmethod
    def get_query_model_params(self) -> Iterator[Parameter]:
        pass

    @abstractmethod
    def get_items_model_params(self) -> Iterator[Parameter]:
        pass

    @abstractmethod
    def fix_query_model(self, num_fixed_layers: int):
        """One of fine-tuning hyperparams is num of fixed layers at a query model

        :param num_fixed_layers: how many layers to fix
        """

    @abstractmethod
    def unfix_query_model(self):
        """Unfix all layers of a query model."""

    @abstractmethod
    def fix_item_model(self, num_fixed_layers: int):
        """One of fine-tuning hyperparams is num of fixed layers at an item model

        :param num_fixed_layers: how many layers to fix
        """

    @abstractmethod
    def unfix_item_model(self):
        """Unfix all layers of an item model."""

    @abstractmethod
    def forward_query(self, query: Any) -> FloatTensor:
        pass

    @abstractmethod
    def forward_items(self, items: List[Any]) -> FloatTensor:
        pass
