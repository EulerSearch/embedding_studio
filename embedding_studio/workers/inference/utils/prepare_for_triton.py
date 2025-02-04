import logging
import os

import torch

from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.inference_management.triton.model_storage_info import (
    DeployedModelInfo,
    ModelStorageInfo,
)
from embedding_studio.utils.gpu_monitoring import select_device

logger = logging.getLogger(__name__)


def mark_same_query_and_items(query_model_path: str):
    """
    Marks that the query and items models are the same, by creating a flag file.

    :param query_model_path: The path to the query model.
    """
    with open(os.path.join(query_model_path, "same_query"), "w") as f:
        f.write("TRUE")


@torch.no_grad()
def convert_for_triton(
    model: EmbeddingsModelInterface,
    plugin_name: str,
    model_repo: str,
    model_version: int,
    embedding_model_id: str,
    embedding_studio_path: str = "/embedding_studio",
):
    """
    Prepares and deploys a model for use with the Triton Inference Server,
    including dynamic GPU selection. This involves tracing the model
    on a selected GPU, saving the traced model,and generating Triton
    configuration files.

    Args:
        model (EmbeddingsModelInterface): The model interface providing access
                                          to the query and item models.
        plugin_name (str): The name used for creating directories
                           and files for the model.
        model_repo (str): The file path to the repository  where the model
                          versions will be stored.
        model_version (int): The version number of the model to be saved.
        embedding_model_id (str): A unique identifier for the model.
    """
    logger.info(f"Embedding model {embedding_model_id} deployment.")
    query_device = select_device()  # Dynamic GPU selection
    logger.info(f"Query model device: {query_device}")
    # Process and save the query model
    query_model = model.get_query_model().to(query_device)
    query_model.eval()
    query_example_inputs = model.get_query_model_inputs(device=query_device)

    query_model_storage_info = ModelStorageInfo(
        model_repo=model_repo,
        embedding_studio_path=embedding_studio_path,
        deployed_model_info=DeployedModelInfo(
            plugin_name=plugin_name,
            model_type="query",
            embedding_model_id=embedding_model_id,
            version=str(model_version),
        ),
    )
    query_model_manager = model.get_query_model_inference_manager_class()(
        query_model_storage_info
    )
    query_model_manager.save_model(
        query_model, query_example_inputs, named_inputs=model.is_named_inputs
    )

    # Check if the same model is used for both queries and items
    if model.same_query_and_items:
        mark_same_query_and_items(query_model_storage_info.model_path)
    else:
        items_device = select_device()
        logger.info(f"Items model device: {items_device}")
        # Process and save the items model
        items_model = model.get_items_model().to(items_device)
        items_model.eval()

        items_example_inputs = model.get_items_model_inputs(
            device=items_device
        )
        items_model_storage_info = ModelStorageInfo(
            model_repo=model_repo,
            embedding_studio_path=embedding_studio_path,
            deployed_model_info=DeployedModelInfo(
                plugin_name=plugin_name,
                model_type="items",
                embedding_model_id=embedding_model_id,
                version=str(model_version),
            ),
        )
        items_model_manager = model.get_items_model_inference_manager_class()(
            items_model_storage_info
        )
        items_model_manager.save_model(items_model, items_example_inputs)
