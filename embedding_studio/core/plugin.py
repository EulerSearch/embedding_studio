import importlib
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.core.config import settings
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.embeddings.improvement.vectors_adjuster import (
    VectorsAdjuster,
)
from embedding_studio.embeddings.inference.triton.client import (
    TritonClientFactory,
)
from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter
from embedding_studio.embeddings.splitters.no_splitter import NoSplitter
from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.models.clickstream.sessions import SessionWithEvents
from embedding_studio.models.embeddings.models import (
    EmbeddingModelInfo,
    SearchIndexInfo,
)
from embedding_studio.models.plugin import FineTuningBuilder, PluginMeta


class FineTuningMethod(ABC):
    """Base class (plugin) for fine-tuning methods.

    All fine-tuning methods must inherit from this class.
    """

    meta: PluginMeta

    def get_items_splitter(self) -> ItemSplitter:
        """Return an ItemSplitter instance.

        By default, returns an NoSplitter instance.

        :return: An instance of ItemSplitter.
        """
        return NoSplitter()

    @abstractmethod
    def get_data_loader(self) -> DataLoader:
        """Return a DataLoader instance.

        Method that should be implemented by subclasses to provide a
        DataLoader instance.

        :return: An instance of DataLoader.
        """
        raise NotImplementedError("Subclasses must implement get_data_loader")

    @abstractmethod
    def get_query_retriever(self) -> QueryRetriever:
        """Return a QueryRetriever instance.

        Method that should be implemented by subclasses to provide a
        QueryRetriever instance.

        :return: An instance of QueryRetriever.
        """
        raise NotImplementedError(
            "Subclasses must implement get_query_retriever"
        )

    @abstractmethod
    def get_manager(self) -> ExperimentsManager:
        """Return a ExperimentsManager instance.

        Method that should be implemented by subclasses to provide a
        ExperimentsManager instance.

        :return: An instance of ExperimentsManager.
        """
        raise NotImplementedError("Subclasses must implement get_manager")

    @abstractmethod
    def get_inference_client_factory(self) -> TritonClientFactory:
        """Return a TritonClientFactory instance.

        Method that should be implemented by subclasses to provide a
        TritonClientFactory instance.

        :return: An instance of TritonClientFactory.
        """
        raise NotImplementedError(
            "Subclasses must implement get_inference_client_factory"
        )

    @abstractmethod
    def upload_initial_model(self) -> None:
        """Upload the initial model to the items_set.

        Method that should be implemented by subclasses to upload the
        initial model to the items_set.
        """
        raise NotImplementedError(
            "Subclasses must implement upload_initial_model"
        )

    @abstractmethod
    def get_fine_tuning_builder(
        self, clickstream: List[SessionWithEvents]
    ) -> FineTuningBuilder:
        """Return a FineTuningBuilder instance for the fine-tuning process.

        Method that should be implemented by subclasses to provide a
        FineTuningBuilder instance.

        :param clickstream: Collection of user feedback, used to enhance
            the model.
        :return: An instance of FineTuningBuilder used for
            launching the fine-tuning process.
        """
        raise NotImplementedError(
            "Subclasses must implement get_fine_tuning_builder"
        )

    def get_embedding_model_info(self, id: str) -> EmbeddingModelInfo:
        """Return a EmbeddingModelInfo instance.

        :param id: ID of an embedding model in model storage system.
        :return: An instance of EmbeddingModelInfo.
        """
        return EmbeddingModelInfo(name=self.meta.name, id=id)

    @abstractmethod
    def get_search_index_info(self) -> SearchIndexInfo:
        """Return a SearchIndexInfo instance. Define a parameters of vectordb index.

        Method that should be implemented by subclasses to provide a
        SearchIndexInfo instance.
        """
        raise NotImplementedError(
            "Subclasses must implement get_search_index_info"
        )

    @abstractmethod
    def get_vectors_adjuster(self) -> VectorsAdjuster:
        """Return a VectorsAdjuster instance. Define a function of vectors changing procedure.

        Method that should be implemented by subclasses to provide a
        VectorsAdjuster instance.
        """
        raise NotImplementedError(
            "Subclasses must implement get_vectors_adjuster"
        )


class PluginManager:
    """Manages the loading and return of plugins."""

    def __init__(self):
        """Initialize an instance of the PluginManager."""
        self._plugins: Dict[str, FineTuningMethod] = {}

    @property
    def plugin_names(self) -> List[str]:
        return list(self._plugins.keys())

    def get_plugin(self, name: str) -> Optional[FineTuningMethod]:
        """Return a plugin by name.

        :param name: The name of the plugin.
        :return: The plugin instance, or None if not found.
        """
        return self._plugins.get(name)

    def discover_plugins(self, directory: str) -> None:
        """Discover and load plugins from the specified directory.

        :param directory: The directory to search for plugins.
        :return: The directory to search for plugins.
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Directory `{directory}` not found")
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name, _ = os.path.splitext(filename)
                module_path = f"{directory}.{module_name}"
                module = importlib.import_module(module_path)

                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, FineTuningMethod)
                        and obj != FineTuningMethod
                    ):
                        if not hasattr(obj, "meta"):
                            raise ValueError(
                                "Plugin meta information is not defined."
                            )
                        if obj.meta.name in settings.INFERENCE_USED_PLUGINS:
                            self._register_plugin(obj())

    def _register_plugin(self, plugin: FineTuningMethod) -> None:
        """Register a plugin in the manager.

        :param plugin: The plugin instance to register.
        :return: None
        """
        if isinstance(plugin, FineTuningMethod):
            self._plugins[plugin.meta.name] = plugin
        else:
            raise ValueError(
                "Only instances of FineTuningMethod can be registered."
            )
