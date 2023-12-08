import logging
from typing import Dict, Iterator, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import FloatTensor
from torch.nn import Parameter

from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.embeddings.models.pooler_output import (
    PassPoolerOutputLayer,
)

logger = logging.getLogger(__name__)


class TextToImageCLIPModel(EmbeddingsModelInterface):
    def __init__(self, clip_model: SentenceTransformer):
        """Wrapper to SentenceTransformer CLIP model.
        Usage: model = TextToImageCLIPModel(SentenceTransformer('clip-ViT-B-32'))

        :param clip_model: clip model from SentenceTransformer package
        :type clip_model: SentenceTransformer
        """
        super(TextToImageCLIPModel, self).__init__(same_query_and_items=False)
        self.clip_model = clip_model
        self.text_model = torch.nn.Sequential(
            self.clip_model._modules["0"]
            ._modules["model"]
            ._modules["text_model"],
            PassPoolerOutputLayer(),
            self.clip_model._modules["0"]
            ._modules["model"]
            ._modules["text_projection"],
        )

        self.vision_model = torch.nn.Sequential(
            self.clip_model._modules["0"]
            ._modules["model"]
            ._modules["vision_model"],
            PassPoolerOutputLayer(),
            self.clip_model._modules["0"]
            ._modules["model"]
            ._modules["visual_projection"],
        )

    def get_query_model_params(self) -> Iterator[Parameter]:
        return self.text_model.parameters()

    def get_items_model_params(self) -> Iterator[Parameter]:
        return self.vision_model.parameters()

    def fix_query_model(self, num_fixed_layers: int):
        if (
            len(self.text_model._modules["0"].encoder.layers)
            >= num_fixed_layers
        ):
            raise ValueError(
                f"Number of fixed layers ({num_fixed_layers}) >= number "
                f'of existing layers ({len(self.text_model._modules["0"].encoder.layers)})'
            )

        self.text_model._modules["0"].embeddings.requires_grad = False
        for i, attn in enumerate(self.text_model._modules["0"].encoder.layers):
            if i < num_fixed_layers:
                self.text_model._modules["0"].encoder.layers[
                    i
                ].requires_grad = False

    def unfix_query_model(self):
        self.text_model._modules["0"].embeddings.requires_grad = True
        for i, attn in enumerate(self.text_model._modules["0"].encoder.layers):
            self.text_model._modules["0"].encoder.layers[
                i
            ].requires_grad = True

    def fix_item_model(self, num_fixed_layers: int):
        if (
            len(self.vision_model._modules["0"].encoder.layers)
            >= num_fixed_layers
        ):
            raise ValueError(
                f"Number of fixed layers ({num_fixed_layers}) >= number "
                f'of existing layers ({len(self.vision_model._modules["0"].encoder.layers)})'
            )

        self.vision_model._modules["0"].embeddings.requires_grad = False
        for i, attn in enumerate(
            self.vision_model._modules["0"].encoder.layers
        ):
            if i < num_fixed_layers:
                self.vision_model._modules["0"].encoder.layers[
                    i
                ].requires_grad = False

    def unfix_item_model(self):
        self.vision_model._modules["0"].embeddings.requires_grad = True
        for i, attn in enumerate(
            self.vision_model._modules["0"].encoder.layers
        ):
            self.vision_model._modules["0"].encoder.layers[
                i
            ].requires_grad = True

    def tokenize(self, query: str) -> List[Dict]:
        return self.clip_model.tokenize([query])

    def forward_query(self, query: str) -> FloatTensor:
        if len(query) == 0:
            logger.warning("Provided query is empty")

        tokenized = self.tokenize(query)
        return self.text_model.forward(tokenized["input_ids"].to(self.device))

    def forward_items(self, items: List[np.array]) -> FloatTensor:
        if len(items) == 0:
            raise ValueError("items list must not be empty")

        return self.vision_model.forward(torch.stack(items).to(self.device))
