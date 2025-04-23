# Documentation for `PassPoolerOutputLayer`

## Overview
The `PassPoolerOutputLayer` class is a utility module designed for extracting the `pooler_output` from transformer model outputs. It facilitates the seamless integration of transformer models with other PyTorch modules by using `torch.nn.Sequential`. 

## Functionality
The `forward` method of the `PassPoolerOutputLayer` class takes an input object (x) that possesses a `pooler_output` attribute and directly returns this attribute. This approach simplifies the process of retrieving `pooler_output`, enabling the combination of transformer outputs with sequential layers without additional transformation.

## Inheritance
`PassPoolerOutputLayer` inherits from `torch.nn.Module`.

## Parameters
- `x`: An object, typically the output from a transformer model, from which the `pooler_output` attribute is extracted.

## Motivation
Many transformer architectures return output objects that contain various attributes, including `pooler_output`. This class aims to streamline the process of accessing `pooler_output` when combining transformers with other layers in a neural network.

## Usage
### Purpose
The primary purpose of the `PassPoolerOutputLayer` is to pass through the `pooler_output` from a transformer output effortlessly.

### Example
```python
import torch
from transformers import BertModel
from embedding_studio.embeddings.models.utils.pooler_output import PassPoolerOutputLayer

model = BertModel.from_pretrained("bert-base-uncased")
pooler_layer = PassPoolerOutputLayer()
outputs = model(input_ids=torch.tensor([[101, 2003, 102]]))
pooler = pooler_layer(outputs)
print(pooler)
```