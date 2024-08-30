import logging
from typing import Dict, Iterator, List, Optional, Type, Union

import torch
from sentence_transformers import SentenceTransformer
from torch import FloatTensor, Tensor
from torch.nn import Module, Parameter
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel

from embedding_studio.context.app_context import context
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


class E5ModelSimplifiedWrapper(torch.nn.Module):
    def __init__(self, model: Union[XLMRobertaModel, torch.nn.Module]):
        super(E5ModelSimplifiedWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.model.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )

        sum_embeddings = torch.sum(output.last_hidden_state, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled = sum_embeddings / sum_mask.clamp(
            min=1
        )  # Avoid division by zero

        return torch.nn.functional.normalize(pooled, p=2, dim=1)


class TextToTextE5Model(EmbeddingsModelInterface):
    """Wrapper to AutoModel or SentenceTransformer E5 model.
    Usage: model = TextToTextE5Model(SentenceTransformer('intfloat/multilingual-e5-large'))

    :param e5_model: E5 type model, either AutoModel, either SentenceTransformer.
    :param e5_tokenizer: E5 tokenizer. If None, will upload it by name. Don't need to speсify for SentenceTransformer version.
    :param max_length: maximum tokens count being used.
    """

    def __init__(
        self,
        e5_model: Union[AutoModel, SentenceTransformer],
        e5_tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
    ):
        super(TextToTextE5Model, self).__init__(same_query_and_items=True)
        if isinstance(e5_model, AutoModel):
            if e5_tokenizer is None:
                self.e5_tokenizer = context.model_downloader.download_model(
                    model_name=e5_model.name_or_path,
                    download_fn=lambda tn: AutoTokenizer.from_pretrained(tn),
                )

            else:
                self.e5_tokenizer = e5_tokenizer

        elif isinstance(e5_model, SentenceTransformer):
            self.e5_tokenizer = e5_model.tokenizer

        elif isinstance(e5_model, str):
            e5_model = SentenceTransformer(e5_model)
            self.e5_tokenizer = e5_model.tokenizer

        self.e5_model = E5ModelSimplifiedWrapper(
            e5_model._modules["0"]._modules["auto_model"]
        )

        self.max_length = max_length

    def get_query_model(self) -> Module:
        return self.e5_model

    def get_items_model(self) -> Module:
        return self.e5_model

    def get_query_model_params(self) -> Iterator[Parameter]:
        return self.e5_model.parameters()

    def get_items_model_params(self) -> Iterator[Parameter]:
        return self.get_query_model_params()

    @torch.no_grad()
    def get_query_model_inputs(self, device=None) -> Dict[str, Tensor]:
        # Tokenize the text
        inputs = self.e5_tokenizer(
            TEST_INPUT_TEXTS,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Extract the input_ids tensor which will be used as the input to the model
        return inputs.to(device if device else self.device)

    @torch.no_grad()
    def get_items_model_inputs(self, device=None) -> Dict[str, Tensor]:
        return self.get_query_model_input(device)

    def get_query_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        return JitTraceTritonModelStorageManager

    def get_items_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        return JitTraceTritonModelStorageManager

    def fix_query_model(self, num_fixed_layers: int):
        if len(self.e5_model._modules["encoder"].layer) <= num_fixed_layers:
            raise ValueError(
                f"Number of fixed layers ({num_fixed_layers}) >= number "
                f"of existing layers ({len(self.e5_model._modules['encoder'].layer)})"
            )
        self.e5_model._modules["embeddings"].requires_grad = False
        for i, attn in enumerate(self.e5_model._modules["encoder"].layer):
            if i < num_fixed_layers:
                self.e5_model._modules["encoder"].layer[
                    i
                ].requires_grad = False

    def unfix_query_model(self):
        self.e5_model._modules["embeddings"].requires_grad = True
        for i, attn in enumerate(self.e5_model._modules["encoder"].layer):
            self.e5_model._modules["encoder"].layer[i].requires_grad = True

    def fix_item_model(self, num_fixed_layers: int):
        self.fix_query_model(num_fixed_layers)

    def unfix_item_model(self):
        self.unfix_query_model()

    def tokenize(self, query: Union[str, List[str]]) -> List[Dict]:
        return self.e5_tokenizer(
            (
                [
                    query,
                ]
                if isinstance(query, str)
                else query
            ),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def forward_query(self, query: str) -> Union[FloatTensor, Tensor]:
        if len(query) == 0:
            logger.warning("Provided query is empty")

        tokenized = self.tokenize(f"query: {query}")
        return self.e5_model.forward(
            input_ids=tokenized["input_ids"].to(self.device),
            attention_mask=tokenized["attention_mask"].to(self.device),
        )

    def forward_items(self, items: List[str]) -> Union[FloatTensor, Tensor]:
        if len(items) == 0:
            raise ValueError("items list must not be empty")

        tokenized = self.tokenize(items)
        return self.e5_model.forward(
            input_ids=tokenized["input_ids"].to(self.device),
            attention_mask=tokenized["attention_mask"].to(self.device),
        )
