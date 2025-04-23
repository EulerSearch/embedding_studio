from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Type

import pytorch_lightning as pl
from torch import FloatTensor, Tensor
from torch.nn import Parameter

from embedding_studio.inference_management.triton.manager import (
    TritonModelStorageManager,
)


class EmbeddingsModelInterface(pl.LightningModule):
    """Interface for embedding models used in fine-tuning procedures.

    This abstract class provides a unified interface for handling embedding models that work
    with query and item entities, which may be multi-domain. The interface is designed to
    facilitate fine-tuning, parameter access, and inference management of embedding models.

    :param same_query_and_items: Boolean indicating if query and items models are the same model
    """

    def __init__(self, same_query_and_items: bool = False):
        """In search results we have two entities, which could be multi domain: query and search result (item).
        This is the interface we used in fine-tuning procedure.

        :param same_query_and_items: Boolean indicating if query and items models are the same model (default: False)
        """
        super(EmbeddingsModelInterface, self).__init__()
        self.same_query_and_items = same_query_and_items

    @abstractmethod
    def get_query_model_params(self) -> Iterator[Parameter]:
        """Get iterator over parameters of the query model.

        :return: Iterator over the parameters of the query model

        Example implementation:
        ```python
        def get_query_model_params(self) -> Iterator[Parameter]:
            return self.query_model.parameters()
        ```
        """

    @abstractmethod
    def get_items_model_params(self) -> Iterator[Parameter]:
        """Get iterator over parameters of the items model.

        :return: Iterator over the parameters of the items model

        Example implementation:
        ```python
        def get_items_model_params(self) -> Iterator[Parameter]:
            return self.items_model.parameters()
        ```
        """

    @property
    def is_named_inputs(self) -> bool:
        """Property indicating whether the model uses named inputs.

        :return: Boolean indicating if the model expects named inputs

        Example implementation:
        ```python
        @property
        def is_named_inputs(self) -> bool:
            return True  # If model expects inputs like {"input_ids": tensor, "attention_mask": tensor}
        ```
        """
        raise NotImplementedError()

    @abstractmethod
    def get_query_model_inputs(self, device=None) -> Dict[str, Tensor]:
        """Get example inputs for the query model, typically for model tracing.

        :param device: Device to place the tensors on. If None, the model's device will be used.
        :return: Dictionary of input tensors for the query model

        Example implementation:
        ```python
        def get_query_model_inputs(self, device=None) -> Dict[str, Tensor]:
            inputs = self.tokenizer("example query", return_tensors="pt")
            device = device if device else self.device
            return {key: value.to(device) for key, value in inputs.items()}
        ```
        """

    @abstractmethod
    def get_items_model_inputs(self, device=None) -> Dict[str, Tensor]:
        """Get example inputs for the items model, typically for model tracing.

        :param device: Device to place the tensors on. If None, the model's device will be used.
        :return: Dictionary of input tensors for the items model

        Example implementation:
        ```python
        def get_items_model_inputs(self, device=None) -> Dict[str, Tensor]:
            inputs = self.tokenizer("example item", return_tensors="pt")
            device = device if device else self.device
            return {key: value.to(device) for key, value in inputs.items()}
        ```
        """

    @abstractmethod
    def get_query_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        """Get the class for managing query model inference in Triton.

        :return: Class for managing query model inference

        Example implementation:
        ```python
        def get_query_model_inference_manager_class(self) -> Type[TritonModelStorageManager]:
            return JitTraceTritonModelStorageManager
        ```
        """

    @abstractmethod
    def get_items_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        """Get the class for managing items model inference in Triton.

        :return: Class for managing items model inference

        Example implementation:
        ```python
        def get_items_model_inference_manager_class(self) -> Type[TritonModelStorageManager]:
            return JitTraceTritonModelStorageManager
        ```
        """

    @abstractmethod
    def fix_query_model(self, num_fixed_layers: int):
        """Fix a specific number of layers in the query model during fine-tuning.

        This method freezes the specified number of layers by setting their requires_grad
        attribute to False, preventing updates during training.

        :param num_fixed_layers: Number of layers to fix from the bottom of the model

        Example implementation:
        ```python
        def fix_query_model(self, num_fixed_layers: int):
            if len(self.query_model.encoder.layers) <= num_fixed_layers:
                raise ValueError(f"Number of fixed layers ({num_fixed_layers}) >= number of existing layers ({len(self.query_model.encoder.layers)})")

            self.query_model.embeddings.requires_grad = False
            for i in range(num_fixed_layers):
                self.query_model.encoder.layers[i].requires_grad = False
        ```
        """

    @abstractmethod
    def unfix_query_model(self):
        """Unfix all layers of the query model.

        This method enables gradient updates for all layers by setting
        their requires_grad attribute to True.

        Example implementation:
        ```python
        def unfix_query_model(self):
            self.query_model.embeddings.requires_grad = True
            for layer in self.query_model.encoder.layers:
                layer.requires_grad = True
        ```
        """

    @abstractmethod
    def fix_item_model(self, num_fixed_layers: int):
        """Fix a specific number of layers in the item model during fine-tuning.

        This method freezes the specified number of layers by setting their requires_grad
        attribute to False, preventing updates during training.

        :param num_fixed_layers: Number of layers to fix from the bottom of the model

        Example implementation:
        ```python
        def fix_item_model(self, num_fixed_layers: int):
            if len(self.items_model.encoder.layers) <= num_fixed_layers:
                raise ValueError(f"Number of fixed layers ({num_fixed_layers}) >= number of existing layers ({len(self.items_model.encoder.layers)})")

            self.items_model.embeddings.requires_grad = False
            for i in range(num_fixed_layers):
                self.items_model.encoder.layers[i].requires_grad = False
        ```
        """

    @abstractmethod
    def unfix_item_model(self):
        """Unfix all layers of the item model.

        This method enables gradient updates for all layers by setting
        their requires_grad attribute to True.

        Example implementation:
        ```python
        def unfix_item_model(self):
            self.items_model.embeddings.requires_grad = True
            for layer in self.items_model.encoder.layers:
                layer.requires_grad = True
        ```
        """

    @abstractmethod
    def forward_query(self, query: Any) -> FloatTensor:
        """Process query through the query model and return embedding.

        :param query: Query input which could be text, features, or any other input format
        :return: Embedding tensor for the query

        Example implementation:
        ```python
        def forward_query(self, query: str) -> FloatTensor:
            if len(query) == 0:
                logger.warning("Provided query is empty")

            tokenized = self.tokenize(query)
            return self.query_model(
                input_ids=tokenized["input_ids"].to(self.device),
                attention_mask=tokenized["attention_mask"].to(self.device)
            )
        ```
        """

    @abstractmethod
    def forward_items(self, items: List[Any]) -> FloatTensor:
        """Process a list of items through the items model and return embeddings.

        :param items: List of items which could be text, features, or any other input format
        :return: Embedding tensor for the items

        Example implementation:
        ```python
        def forward_items(self, items: List[str]) -> FloatTensor:
            if len(items) == 0:
                raise ValueError("items list must not be empty")

            tokenized = self.tokenize(items)
            return self.items_model(
                input_ids=tokenized["input_ids"].to(self.device),
                attention_mask=tokenized["attention_mask"].to(self.device)
            )
        ```
        """
