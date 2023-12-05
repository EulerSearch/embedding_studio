from typing import Union

import torch
from torch import FloatTensor, Tensor


class PassPoolerOutputLayer(torch.nn.Module):
    """Transformers usually have object as an output.
    But if you want to stack transformer and another model as Sequential.
    You can use this module to pass pooler_output.

    """

    def __init__(self):
        super().__init__()

    def forward(self, x) -> Union[FloatTensor, Tensor]:
        return x.pooler_output
