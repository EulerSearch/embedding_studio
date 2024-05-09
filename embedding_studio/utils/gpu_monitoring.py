import subprocess

import torch

last_used_gpu = -1


def select_gpu():
    """
    Selects the most suitable GPU based on current load.

    :return: The ID of the selected GPU.
    """
    try:
        return get_least_loaded_gpu()
    except Exception as e:
        print(f"Failed to run nvidia-smi: {e}")
        # Fallback to round-robin selection if nvidia-smi fails
        return get_next_gpu()


def get_least_loaded_gpu():
    """
    Retrieves the GPU with the lowest memory usage.

    :return: The index of the GPU with the minimum memory usage.
    """
    smi_output = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        .decode()
        .strip()
    )
    gpu_memory = [int(x) for x in smi_output.split("\n")]
    least_loaded_gpu = gpu_memory.index(min(gpu_memory))
    return least_loaded_gpu


def get_next_gpu():
    """
    Selects the next GPU in a round-robin fashion.

    :return: The ID of the next GPU to use.
    """
    global last_used_gpu
    num_gpus = torch.cuda.device_count()
    last_used_gpu = (last_used_gpu + 1) % num_gpus
    return last_used_gpu


def select_device():
    """
    Selects the most suitable device (GPU or CPU).

    :return: The selected device, either 'cuda' or 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{select_gpu()}")
    else:
        return torch.device("cpu")
