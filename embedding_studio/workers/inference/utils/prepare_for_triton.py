import logging
import os

import torch

from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.inference_management.triton.model_storage_info import (
    ModelStorageInfo,
)
from embedding_studio.utils.gpu_monitoring import select_device

logger = logging.getLogger(__name__)


def write_experiment_data(
    model_path: str, experiment_id: str, deployment_flag: str
):
    """
    Writes experiment ID and deployment flag to the model directory.

    :param model_path: Path to the model directory.
    :param experiment_id: The experiment identifier.
    :param deployment_flag: Flag indicating if the deployment should be marked green.
    """
    with open(os.path.join(model_path, "experiment_id"), "w") as f:
        f.write(experiment_id)
    with open(os.path.join(model_path, "green_deployment.flag"), "w") as f:
        f.write(deployment_flag)


def mark_same_query_and_items(model_repo: str, plugin_name: str):
    """
    Marks that the query and items models are the same, by creating a flag file.

    :param model_repo: The repository path where the model is stored.
    :param plugin_name: The plugin name under which the model is stored.
    """
    with open(
        os.path.join(model_repo, f"{plugin_name}.query", "same_query"), "w"
    ) as f:
        f.write("TRUE")


@torch.no_grad()
def convert_for_triton(
    model: EmbeddingsModelInterface,
    plugin_name: str,
    model_repo: str,
    model_version: int,
    experiment_id: str,
    embedding_studio_path: str = "/embedding_studio",
):
    """
    Prepares and deploys a model for use with the Triton Inference Server, including dynamic GPU selection.
    This involves tracing the model on a selected GPU, saving the traced model, and generating Triton configuration files.

    Args:
        model (EmbeddingsModelInterface): The model interface providing access to the query and item models.
        plugin_name (str): The name used for creating directories and files for the model.
        model_repo (str): The file path to the repository where the model versions will be stored.
        model_version (int): The version number of the model to be saved.
        experiment_id (str): A unique identifier for the experiment.
    """
    query_device = select_device()  # Dynamic GPU selection
    logger.info(f"Query model device: {query_device}")
    # Process and save the query model
    query_model = model.get_query_model().to(query_device)
    query_model.eval()
    query_input_name, query_example_input = model.get_query_model_input()
    query_example_input = query_example_input.to(query_device)
    query_model_storage_info = ModelStorageInfo(
        model_repo=model_repo,
        plugin_name=plugin_name,
        model_type="query",
        version=str(model_version),
        embedding_studio_path=embedding_studio_path,
    )
    query_model_manager = model.get_query_model_inference_manager_class()(
        query_model_storage_info
    )
    query_model_manager.save_model(
        query_model, query_example_input, query_input_name
    )

    write_experiment_data(
        query_model_storage_info.model_version_path, experiment_id, "TRUE"
    )

    # Check if the same model is used for both queries and items
    if model.same_query_and_items:
        mark_same_query_and_items(model_repo, plugin_name)
    else:
        items_device = select_device()
        logger.info(f"Items model device: {items_device}")
        # Process and save the items model
        items_model = model.get_items_model().to(items_device)
        items_model.eval()

        items_input_name, items_example_input = model.get_items_model_input()
        items_example_input = items_example_input.to(items_device)
        items_model_storage_info = ModelStorageInfo(
            model_repo=model_repo,
            plugin_name=plugin_name,
            model_type="items",
            version=str(model_version),
            embedding_studio_path=embedding_studio_path,
        )
        items_model_manager = model.get_items_model_inference_manager_class()(
            items_model_storage_info
        )
        items_model_manager.save_model(
            items_model, items_example_input, items_input_name
        )
        write_experiment_data(
            items_model_storage_info.model_version_path, experiment_id, "TRUE"
        )
