import logging
import os
from typing import Dict, Iterator, List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch import FloatTensor, Tensor
from torch.nn import Module, Parameter
from torchvision import transforms
from torchvision.transforms import Resize

from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.embeddings.models.pooler_output import (
    PassPoolerOutputLayer,
)
from embedding_studio.inference_management.triton.jit_trace_manager import (
    JitTraceTritonModelStorageManager,
)
from embedding_studio.inference_management.triton.manager import (
    TritonModelStorageManager,
)

logger = logging.getLogger(__name__)


class TextToImageCLIPModel(EmbeddingsModelInterface):
    def __init__(self, clip_model: SentenceTransformer):
        """Wrapper to SentenceTransformer CLIP model.
        Usage: model = TextToImageCLIPModel(SentenceTransformer('clip-ViT-B-32'))

        :param clip_model: clip model from SentenceTransformer package
        """
        super(TextToImageCLIPModel, self).__init__(same_query_and_items=False)
        self.tokenizer = clip_model[0].processor.tokenizer

        self.text_model = torch.nn.Sequential(
            clip_model._modules["0"]._modules["model"]._modules["text_model"],
            PassPoolerOutputLayer(),
            clip_model._modules["0"]
            ._modules["model"]
            ._modules["text_projection"],
        )

        self.vision_model = torch.nn.Sequential(
            clip_model._modules["0"]
            ._modules["model"]
            ._modules["vision_model"],
            PassPoolerOutputLayer(),
            clip_model._modules["0"]
            ._modules["model"]
            ._modules["visual_projection"],
        )

    def get_query_model(self) -> Module:
        return self.text_model

    def get_items_model(self) -> Module:
        return self.vision_model

    def get_query_model_params(self) -> Iterator[Parameter]:
        return self.text_model.parameters()

    def get_items_model_params(self) -> Iterator[Parameter]:
        return self.vision_model.parameters()

    @torch.no_grad()
    def get_query_model_input(self) -> Tuple[str, Tensor]:
        # Define an example text
        text = "Example text to be tokenized and input into the model."

        # Tokenize the text
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        # Extract the input_ids tensor which will be used as the input to the model
        return "input_ids", inputs["input_ids"]

    @torch.no_grad()
    def get_items_model_input(
        self, image: Optional[Image.Image] = None
    ) -> Tuple[str, Tensor]:
        if image is None:
            # This comment underscores the necessity of providing a real image as example_input during tracing for Triton,
            # which is pivotal for model compilation using torch.jit.trace(model, example_input).
            # Throughout this process, the torch.jit module records solely the active paths in the computation graph.
            # Supplying irrelevant input data, like a black square, poses the risk of creating a suboptimal or erroneous model.
            # Consequently, the inclusion of a real image guarantees the precise representation of pertinent paths and
            # dependencies within the computation graph, thereby bolstering the fidelity and performance of the resultant model.
            image_path = os.path.join(
                os.path.dirname(__file__), "image-for-tracing.png"
            )
            image = Image.open(image_path).convert("RGB")

        # Resize the image and prepare for model input
        resize_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                Resize((224, 224)),  # Resize the image to 224x224
                transforms.Normalize(
                    # These values represent the mean and standard deviation used for normalizing the input data in a ResNet model.
                    # Normalization helps ensure optimal performance by scaling the input data appropriately before passing it through the model.
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        image = resize_transform(image)

        return "pixel_values", image.unsqueeze(0)

    def get_query_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        return JitTraceTritonModelStorageManager

    def get_items_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        return JitTraceTritonModelStorageManager

    def fix_query_model(self, num_fixed_layers: int):
        if (
            len(self.text_model._modules["0"].encoder.layers)
            <= num_fixed_layers
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
            <= num_fixed_layers
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
        return self.tokenizer(
            [query],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

    def forward_query(self, query: str) -> FloatTensor:
        if len(query) == 0:
            logger.warning("Provided query is empty")

        tokenized = self.tokenize(query)
        return self.text_model.forward(tokenized["input_ids"].to(self.device))

    def forward_items(self, items: List[np.array]) -> FloatTensor:
        if len(items) == 0:
            raise ValueError("items list must not be empty")

        return self.vision_model.forward(torch.stack(items).to(self.device))
