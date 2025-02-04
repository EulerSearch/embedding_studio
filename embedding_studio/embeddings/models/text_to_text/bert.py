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
    """Wrapper for BERT models using AutoModel.
    Usage: model = TextToBertModel(AutoModel.from_pretrained('bert-base-uncased'))

    This implementation expects a BERT model with hidden size of 384 dimensions.

    :param bert_model: BERT model from AutoModel with hidden_size=384
    :param bert_tokenizer: BERT tokenizer. If None, will upload it by name
    :param max_length: maximum tokens count being used
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
        return self.model

    def get_items_model(self) -> Module:
        return self.model

    def get_query_model_params(self) -> Iterator[Parameter]:
        return self.model.parameters()

    def get_items_model_params(self) -> Iterator[Parameter]:
        return self.get_query_model_params()

    @property
    def is_named_inputs(self) -> bool:
        return True

    @torch.no_grad()
    def get_query_model_inputs(self, device=None) -> Dict[str, Tensor]:
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
        return self.get_query_model_inputs(device)

    def get_query_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        return JitTraceTritonModelStorageManager

    def get_items_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        return JitTraceTritonModelStorageManager

    def fix_query_model(self, num_fixed_layers: int):
        if len(self.bert_model.encoder.layer) <= num_fixed_layers:
            raise ValueError(
                f"Number of fixed layers ({num_fixed_layers}) >= number "
                f"of existing layers ({len(self.bert_model.encoder.layer)})"
            )
        self.bert_model.embeddings.requires_grad = False
        for i in range(num_fixed_layers):
            self.bert_model.encoder.layer[i].requires_grad = False

    def unfix_query_model(self):
        self.bert_model.embeddings.requires_grad = True
        for layer in self.bert_model.encoder.layer:
            layer.requires_grad = True

    def fix_item_model(self, num_fixed_layers: int):
        self.fix_query_model(num_fixed_layers)

    def unfix_item_model(self):
        self.unfix_query_model()

    def tokenize(self, query: Union[str, List[str]]) -> Dict:
        return self.bert_tokenizer(
            [query] if isinstance(query, str) else query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def forward_query(self, query: str) -> Union[FloatTensor, Tensor]:
        if len(query) == 0:
            logger.warning("Provided query is empty")

        tokenized = self.tokenize(query)
        return self.model.forward(
            input_ids=tokenized["input_ids"].to(self.device),
            attention_mask=tokenized["attention_mask"].to(self.device),
        )

    def forward_items(self, items: List[str]) -> Union[FloatTensor, Tensor]:
        if len(items) == 0:
            raise ValueError("items list must not be empty")

        tokenized = self.tokenize(items)
        return self.model.forward(
            input_ids=tokenized["input_ids"].to(self.device),
            attention_mask=tokenized["attention_mask"].to(self.device),
        )
