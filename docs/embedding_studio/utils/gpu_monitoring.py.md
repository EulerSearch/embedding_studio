## Documentation for GPU Selection Methods

### `select_gpu`

#### Functionality
This function selects the most suitable GPU based on the current load. It first tries to find the GPU with the lowest memory usage. If that fails, it falls back to a round-robin selection.

#### Parameters
- None

#### Usage
**Purpose**: Choose a GPU for computation based on current load and memory usage.

#### Example
```python
import torch
gpu_id = select_gpu()
device = torch.device(f"cuda:{gpu_id}")
```

---

### `get_least_loaded_gpu`

#### Functionality
Retrieves the GPU index with the lowest memory used. This function fetches GPU memory usage by executing nvidia-smi and returns the GPU ID that has the least memory load.

#### Parameters
This function does not take any parameters.

#### Usage
Use this function to select an optimal GPU for compute tasks. It is particularly useful when multiple GPUs are available.

#### Example
```python
gpu_index = get_least_loaded_gpu()
print(f"Selected GPU: {gpu_index}")
```

---

### `get_next_gpu`

#### Functionality
This function cycles through available GPUs using a round-robin algorithm. It maintains a global counter and returns the next GPU ID by incrementing the last used GPU index and wrapping around if needed.

#### Parameters
This function does not take any parameters.

#### Return Value
- Returns an integer representing the ID of the next GPU to use.

#### Usage
Call this function to distribute workloads evenly across multiple GPUs in a cyclic manner.

#### Example
```python
import torch
from embedding_studio.utils.gpu_monitoring import get_next_gpu

gpu_id = get_next_gpu()
print("Using GPU:", gpu_id)
```

---

### `select_device`

#### Functionality
Selects the most suitable device for computation. This function checks if a GPU is available and returns a torch.device instance. It selects a GPU with the lowest load if available; otherwise, it falls back to the CPU.

#### Parameters
This function does not require any parameters.

#### Usage
**Purpose**: To automatically choose the best computation device for deep learning tasks.

#### Example
```python
import torch
from embedding_studio.utils.gpu_monitoring import select_device

device = select_device()
print("Using device:", device)
```