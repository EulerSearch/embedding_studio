from typing import Dict, List, Union

INITIAL_PARAMS: Dict[str, List[Union[int, float]]] = {
    "num_fixed_layers": [5, 6, 7, 8],
    "query_lr": [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7],
    "items_lr": [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7],
    "query_weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "items_weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "margin": [0.01, 0.025, 0.05],
}
