import importlib
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from embedding_studio.models.clickstream.sessions import SessionWithEvents
from embedding_studio.models.plugin import FineTuningBuilder, PluginMeta


class FineTuningMethod(ABC):
    """Base class (plugin) for fine-tuning methods.

    All fine-tuning methods must inherit from this class.
    """

    meta: PluginMeta

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


class PluginManager:
    """Manages the loading and return of plugins."""

    def __init__(self):
        """Initialize an instance of the PluginManager."""
        self._plugins: Dict[str, FineTuningMethod] = {}

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
