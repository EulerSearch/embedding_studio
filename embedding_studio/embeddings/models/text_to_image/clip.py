import logging
import os
from typing import Dict, Iterator, List, Optional, Type

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
from embedding_studio.embeddings.models.utils.pooler_output import (
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
    """Wrapper for SentenceTransformer CLIP model to create embeddings for text-to-image search.

    This class implements the EmbeddingsModelInterface for CLIP models from the SentenceTransformer package.
    It provides separate text and vision models for query and item processing, allowing text queries
    to be matched with image items in a shared embedding space.

    Usage example:
    ```
    embedding_model = TextToImageCLIPModel(SentenceTransformer('clip-ViT-B-32'))
    ```

    :param clip_model: A CLIP model from the SentenceTransformer package
    """

    def __init__(self, clip_model: SentenceTransformer):
        """Initialize the TextToImageCLIPModel with a CLIP SentenceTransformer model.

        Extracts the text and vision components from the provided CLIP model and configures them
        for separate query (text) and item (image) processing.

        :param clip_model: A CLIP model from the SentenceTransformer package
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
        """Get the text model used for processing queries.

        :return: The text model component
        """
        return self.text_model

    def get_items_model(self) -> Module:
        """Get the vision model used for processing image items.

        :return: The vision model component
        """
        return self.vision_model

    def get_query_model_params(self) -> Iterator[Parameter]:
        """Get iterator over parameters of the text model.

        :return: Iterator over the parameters of the text model
        """
        return self.text_model.parameters()

    def get_items_model_params(self) -> Iterator[Parameter]:
        """Get iterator over parameters of the vision model.

        :return: Iterator over the parameters of the vision model
        """
        return self.vision_model.parameters()

    @property
    def is_named_inputs(self) -> bool:
        """Determine if the model uses named inputs.

        CLIP models do not use named inputs in the traditional sense, as the text and vision
        components expect different input formats.

        :return: False since the model doesn't use a consistent named input structure
        """
        return False

    @torch.no_grad()
    def get_query_model_inputs(self, device=None) -> Dict[str, Tensor]:
        """Get example inputs for the text model, typically for model tracing.

        Creates sample tokenized text input to be used for the text model.

        :param device: Device to place the tensors on. If None, the model's device will be used.
        :return: Dictionary with input_ids tensor for the text model
        """
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
        return {
            "input_ids": inputs["input_ids"].to(
                device if device else self.device
            )
        }

    @torch.no_grad()
    def get_items_model_inputs(
        self, image: Optional[Image.Image] = None, device=None
    ) -> Dict[str, Tensor]:
        """Get example inputs for the vision model, typically for model tracing.

        Creates sample image input to be used for the vision model. If no image is provided,
        it loads a default image from the package.

        :param image: Optional PIL Image to use as input. If None, a default image will be loaded.
        :param device: Device to place the tensors on. If None, the model's device will be used.
        :return: Dictionary with pixel_values tensor for the vision model
        """
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

        return {
            "pixel_values": image.unsqueeze(0).to(
                device if device else self.device
            )
        }

    def get_query_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        """Get the class for managing text model inference in Triton.

        :return: JitTraceTritonModelStorageManager class for text model inference
        """
        return JitTraceTritonModelStorageManager

    def get_items_model_inference_manager_class(
        self,
    ) -> Type[TritonModelStorageManager]:
        """Get the class for managing vision model inference in Triton.

        :return: JitTraceTritonModelStorageManager class for vision model inference
        """
        return JitTraceTritonModelStorageManager

    def fix_query_model(self, num_fixed_layers: int):
        """Fix a specific number of layers in the text model during fine-tuning.

        This method freezes the embeddings and the specified number of encoder layers by setting
        their requires_grad attribute to False, preventing updates during training.

        :param num_fixed_layers: Number of layers to fix from the bottom of the text model
        """
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
        """Unfix all layers of the text model.

        This method enables gradient updates for all layers by setting
        their requires_grad attribute to True.
        """
        self.text_model._modules["0"].embeddings.requires_grad = True
        for i, attn in enumerate(self.text_model._modules["0"].encoder.layers):
            self.text_model._modules["0"].encoder.layers[
                i
            ].requires_grad = True

    def fix_item_model(self, num_fixed_layers: int):
        """Fix a specific number of layers in the vision model during fine-tuning.

        This method freezes the embeddings and the specified number of encoder layers by setting
        their requires_grad attribute to False, preventing updates during training.

        :param num_fixed_layers: Number of layers to fix from the bottom of the vision model
        """
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
        """Unfix all layers of the vision model.

        This method enables gradient updates for all layers by setting
        their requires_grad attribute to True.
        """
        self.vision_model._modules["0"].embeddings.requires_grad = True
        for i, attn in enumerate(
            self.vision_model._modules["0"].encoder.layers
        ):
            self.vision_model._modules["0"].encoder.layers[
                i
            ].requires_grad = True

    def tokenize(self, query: str) -> List[Dict]:
        """Tokenize a text query for processing by the text model.

        :param query: Text query to tokenize
        :return: Tokenized output as dictionary with tensors
        """
        return self.tokenizer(
            [query],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

    def forward_query(self, query: str) -> FloatTensor:
        """Process a text query through the text model and return embedding.

        :param query: Text query to encode
        :return: Embedding tensor for the text query
        """
        if len(query) == 0:
            logger.warning("Provided query is empty")

        tokenized = self.tokenize(query).to(self.device)
        return self.text_model.forward(tokenized["input_ids"])

    def forward_items(self, items: List[np.array]) -> FloatTensor:
        """Process a list of image tensors through the vision model and return embeddings.

        :param items: List of image tensors to encode
        :return: Embedding tensor for the images
        """
        if len(items) == 0:
            raise ValueError("items list must not be empty")

        return self.vision_model.forward(torch.stack(items).to(self.device))
