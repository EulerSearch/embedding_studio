import logging
from typing import Dict, Iterator, List, Optional, Union

from sentence_transformers import SentenceTransformer
from torch import FloatTensor, Tensor
from torch.nn import Parameter
from transformers import AutoModel, AutoTokenizer

from embedding_studio.embeddings.models.average_pool import average_pool
from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)

logger = logging.getLogger(__name__)


class TextToTextE5Model(EmbeddingsModelInterface):
    """Wrapper to AutoModel or SentenceTransformer E5 model.
    Usage: model = TextToImageCLIPModel(SentenceTransformer('intfloat/multilingual-e5-large'))

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
            self.e5_model = e5_model
            if e5_tokenizer is None:
                self.e5_tokenizer = AutoTokenizer.from_pretrained(
                    e5_model.name_or_path
                )
            else:
                self.e5_tokenizer = e5_tokenizer
        else:
            pass

        self.max_length = max_length

    def get_query_model_params(self) -> Iterator[Parameter]:
        return self.e5_model.parameters()

    def get_items_model_params(self) -> Iterator[Parameter]:
        return self.get_query_model_params()

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
            [
                query,
            ]
            if isinstance(query, str)
            else query,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def forward_query(self, query: str) -> Union[FloatTensor, Tensor]:
        if len(query) == 0:
            logger.warning("Provided query is empty")

        tokenized = self.tokenize(f"query: {query}")
        return average_pool(
            self.e5_model(**tokenized).last_hidden_state,
            tokenized["attention_mask"],
        )

    def forward_items(self, items: List[str]) -> Union[FloatTensor, Tensor]:
        if len(items) == 0:
            raise ValueError("items list must not be empty")

        tokenized = self.tokenize(items)
        return average_pool(
            self.e5_model(**tokenized).last_hidden_state,
            tokenized["attention_mask"],
        )
