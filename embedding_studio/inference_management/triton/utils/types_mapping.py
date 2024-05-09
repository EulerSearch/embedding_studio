import torch


def pytorch_dtype_to_triton_dtype(torch_dtype: torch.dtype) -> str:
    """
    Convert PyTorch data types to Triton Inference Server data types.

    :param torch_dtype: The PyTorch data type to convert.
    :return: Corresponding Triton data type as a string.
    """
    mapping = {
        torch.float32: "TYPE_FP32",
        torch.float16: "TYPE_FP16",
        torch.int32: "TYPE_INT32",
        torch.int64: "TYPE_INT64",
        torch.int8: "TYPE_INT8",
        torch.uint8: "TYPE_UINT8",
        torch.bool: "TYPE_BOOL",
    }
    return mapping.get(torch_dtype, "TYPE_FP32")  # Default to FP32 if unknown
