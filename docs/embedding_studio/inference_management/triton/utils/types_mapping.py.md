# Documentation for `pytorch_dtype_to_triton_dtype`

## Overview

The `pytorch_dtype_to_triton_dtype` function converts a PyTorch data type to its equivalent Triton Inference Server data type. It maps common types like `torch.float32` and `torch.int64` to the corresponding Triton strings.

## Parameters

- **torch_dtype**: The PyTorch data type that needs to be converted.

## Return Value

The function returns a string that represents the Triton data type. If an unknown type is provided, it defaults to "TYPE_FP32".

## Usage

This function can be utilized to ensure compatibility when interfacing with a Triton Inference Server by mapping PyTorch dtypes to appropriate Triton types.

### Example

```python
import torch
from embedding_studio.inference_management.triton.utils.types_mapping import pytorch_dtype_to_triton_dtype

dtype_str = pytorch_dtype_to_triton_dtype(torch.float32)
print(dtype_str)  # Output: TYPE_FP32
```