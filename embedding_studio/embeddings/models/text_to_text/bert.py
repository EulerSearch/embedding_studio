import logging
from typing import Dict, Iterator, List, Optional, Type, Union

import torch
from torch import FloatTensor, Tensor
from torch.nn import Module, Parameter
from transformers import AutoModel, AutoTokenizer, BertModel

from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.embeddings.models.text_to_text.test_inputs import (
    TEST_INPUT_TEXTS,
)
from embedding_studio.inference_management.triton.jit_trace_manager import (
    JitTraceTritonModelStorageManager,
)
from embedding_studio.inference_management.triton.manager import (
    TritonModelStorageManager,
)

logger = logging.getLogger(__name__)


class BertModelSimplifiedWrapper(torch.nn.Module):
    def __init__(self, model: Union[BertModel, torch.nn.Module]):
        super(BertModelSimplifiedWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Use BERT's pooler output which applies a linear layer and tanh to the first token
        embedding_output = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output

        return embedding_output


class TextToTextBertModel(EmbeddingsModelInterface):
    """Wrapper for BERT models using AutoModel from transformers.

    This class implements the EmbeddingsModelInterface for BERT-based models,
    providing functionality for text-to-text search with embeddings. It wraps
    AutoModel BERT models and their tokenizers for generating text embeddings.

    Usage example:
    ```
    model = TextToBertModel(AutoModel.from_pretrained('bert-base-uncased'))
    ```

    This implementation expects a BERT model with hidden size of 384 dimensions.

    :param bert_model: BERT model from AutoModel or a string model name
    :param bert_tokenizer: BERT tokenizer from AutoTokenizer. If None, will load by model name
    :param max_length: Maximum tokens count to use for tokenization
    """

    def __init__(
        self,
        bert_model: Union[str, AutoModel],
        bert_tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
    ):
        super(TextToTextBertModel, self).__init__(same_query_and_items=True)

        if isinstance(bert_model, str):
            self.bert_model = AutoModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        if bert_tokenizer is None:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                self.bert_model.config.name_or_path
            )
        else:
            self.bert_tokenizer = bert_tokenizer

        self.model = BertModelSimplifiedWrapper(self.bert_model)
        self.max_length = max_length

    def get_query_model(self) -> Module:
        """Get the model used for processing queries.

        Since query and items use the same model, this returns the wrapped BERT model.

        :return: The query model component
        """
        return self.model

    def get_items_model(self) -> Module:
        """Get the model used for processing items.

        Since query and items use the same model, this returns the wrapped BERT model.

        :return: The items model component
        """
        return self.model

    def get_query_model_params(self) -> Iterator[Parameter]:
        """Get iterator over parameters of the query model.

        :return: Iterator over the parameters of the query model
        """
        return self.model.parameters()

    def get_items_model_params(self) -> Iterator[Parameter]:
        """Get iterator over parameters of the items model.

        Since query and items use the same model, this returns the same parameters as get_query_model_params.

        :return: Iterator over the parameters of the items model
        """
        return self.get_query_model_params()

    @property
    def is_named_inputs(self) -> bool:
        """Determine if the model uses named inputs.

        BERT models require named inputs for input_ids and attention_mask.

        :return: True as the model expects named inputs
        """
        return True

    @torch.no_grad()
    def get_query_model_inputs(self, device=None) -> Dict[str, Tensor]:
        """Get example inputs for the query model, typically for model tracing.

        Creates sample tokenized text inputs using TEST_INPUT_TEXTS.

        :param device: Device to place the tensors on. If None, the model's device will be used
        :return: Dictionary with input_ids and attention_mask tensors for the model
        """
        inputs = self.bert_tokenizer(
            TEST_INPUT_TEXTS,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        inputs = {
            key: value
            for key, value in inputs.items()
            if key in ["input_ids", "attention_mask"]
        }
        # Move each tensor to the specified device
        device = device if device else self.device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        return inputs

    @torch.no_grad()
    def get_items_model_inputs(self, device=None) -> Dict[str, Tensor]:
        """Get example inputs for the items model, typically for model tracing.

        Since query and items use the same model, this returns the same inputs as get_query_model_inputs.

        :param device: Device to place the tensors on. If None, the model's device will be used
        :return: Dictionary with input_ids and attention_mask tensors for the model
        """
        return self.get_query_model_inputs(device)

    def get_query_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        """Get the class for managing query model inference in Triton.

        :return: JitTraceTritonModelStorageManager class for model inference
        """
        return JitTraceTritonModelStorageManager

    def get_items_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        """Get the class for managing items model inference in Triton.

        :return: JitTraceTritonModelStorageManager class for model inference
        """
        return JitTraceTritonModelStorageManager

    def fix_query_model(self, num_fixed_layers: int):
        """Fix a specific number of layers in the query model during fine-tuning.

        This method freezes the embeddings and the specified number of encoder layers by setting
        their requires_grad attribute to False, preventing updates during training.

        :param num_fixed_layers: Number of layers to fix from the bottom of the model
        """
        if len(self.bert_model.encoder.layer) <= num_fixed_layers:
            raise ValueError(
                f"Number of fixed layers ({num_fixed_layers}) >= number "
                f"of existing layers ({len(self.bert_model.encoder.layer)})"
            )
        self.bert_model.embeddings.requires_grad = False
        for i in range(num_fixed_layers):
            self.bert_model.encoder.layer[i].requires_grad = False

    def unfix_query_model(self):
        """Unfix all layers of the query model.

        This method enables gradient updates for all layers by setting
        their requires_grad attribute to True.
        """
        self.bert_model.embeddings.requires_grad = True
        for layer in self.bert_model.encoder.layer:
            layer.requires_grad = True

    def fix_item_model(self, num_fixed_layers: int):
        """Fix a specific number of layers in the item model during fine-tuning.

        Since query and items use the same model, this calls fix_query_model.

        :param num_fixed_layers: Number of layers to fix from the bottom of the model
        """
        self.fix_query_model(num_fixed_layers)

    def unfix_item_model(self):
        """Unfix all layers of the item model.

        Since query and items use the same model, this calls unfix_query_model.
        """
        self.unfix_query_model()

    def tokenize(self, query: Union[str, List[str]]) -> Dict:
        """Tokenize a text query or list of queries for processing by the model.

        :param query: Text query or list of queries to tokenize
        :return: Tokenized output as dictionary with tensors
        """
        return self.bert_tokenizer(
            [query] if isinstance(query, str) else query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def forward_query(self, query: str) -> Union[FloatTensor, Tensor]:
        """Process a text query through the model and return embedding.

        :param query: Text query to encode
        :return: Embedding tensor for the text query
        """
        if len(query) == 0:
            logger.warning("Provided query is empty")

        tokenized = self.tokenize(query)
        return self.model.forward(
            input_ids=tokenized["input_ids"].to(self.device),
            attention_mask=tokenized["attention_mask"].to(self.device),
        )

    def forward_items(self, items: List[str]) -> Union[FloatTensor, Tensor]:
        """Process a list of text items through the model and return embeddings.

        :param items: List of text items to encode
        :return: Embedding tensor for the text items
        """
        if len(items) == 0:
            raise ValueError("items list must not be empty")

        tokenized = self.tokenize(items)
        return self.model.forward(
            input_ids=tokenized["input_ids"].to(self.device),
            attention_mask=tokenized["attention_mask"].to(self.device),
        )
