import importlib
import os
from typing import Any, Dict, Optional

from embedding_studio.models.plugin import FineTuningBuilder, PluginMeta


class FineTuningMethod:
    """
    Base class (plugin) for fine-tuning methods. All fine-tuning methods must
    inherit from this class.
    """

    meta: PluginMeta

    def get_fine_tuning_builder(
        self, clickstream: Dict[str, Any]
    ) -> FineTuningBuilder:
        """
        Method that should be implemented by subclasses to provide a
        FineTuningBuilder instance.

        Args:
            clickstream: A collection of user feedback, used to enhance
            the model.

        Returns:
            FineTuningBuilder: An instance of FineTuningBuilder used for
            launching the fine-tuning process.
        """
        raise NotImplementedError(
            "Subclasses must implement get_fine_tuning_builder"
        )


class PluginManager:
    """
    Manages the loading and return of plugins.
    """

    def __init__(self):
        """
        Initialize an instance of the PluginManager.
        """
        self._plugins: Dict[str, FineTuningMethod] = {}

    def get_plugin(self, name: str) -> Optional[FineTuningMethod]:
        """
        Return a plugin by name.

        Args:
            name (str): The name of the plugin.

        Returns:
            Optional[FineTuningMethod]: The plugin instance, or None if not
            found.
        """
        return self._plugins.get(name)

    def discover_plugins(self, directory: str) -> None:
        """
        Discover and load plugins from the specified directory.

        Args:
            directory (str): The directory to search for plugins.
        """
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
        """
        Register a plugin in the manager.

        Args:
            plugin (FineTuningMethod): The plugin instance to register.
        """
        if isinstance(plugin, FineTuningMethod):
            self._plugins[plugin.meta.name] = plugin
        else:
            raise ValueError(
                "Only instances of FineTuningMethod can be registered."
            )
