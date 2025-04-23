from embedding_studio.context.app_context import context
from embedding_studio.core.plugin import (
    CategoriesFineTuningMethod,
    FineTuningMethod,
)
from embedding_studio.vectordb.vectordb import VectorDb


def is_basic_plugin(plugin: FineTuningMethod) -> bool:
    """
    Check if a plugin is a basic FineTuningMethod (but not a CategoriesFineTuningMethod).

    This function determines if the plugin is a standard fine-tuning method without
    additional categorization capabilities.

    :param plugin: The plugin to check
    :return: True if the plugin is a basic FineTuningMethod, False otherwise
    """
    return issubclass(plugin.__class__, FineTuningMethod) and not issubclass(
        plugin.__class__, CategoriesFineTuningMethod
    )


def is_categories_plugin(plugin: FineTuningMethod) -> bool:
    """
    Check if a plugin is a CategoriesFineTuningMethod.

    This function determines if the plugin supports category-based fine-tuning.

    :param plugin: The plugin to check
    :return: True if the plugin is a CategoriesFineTuningMethod, False otherwise
    """
    return issubclass(plugin.__class__, CategoriesFineTuningMethod)


def get_vectordb(plugin: FineTuningMethod) -> VectorDb:
    """
    Get the appropriate vector database for a given plugin.

    This function selects the correct vector database based on the plugin type:
    - Basic plugins use the standard vector database
    - Categories plugins use the categories vector database

    :param plugin: The plugin to get the vector database for
    :return: The appropriate VectorDb instance for the plugin
    """
    if is_basic_plugin(plugin):
        return context.vectordb

    elif is_categories_plugin(plugin):
        return context.categories_vectordb

    else:
        raise ValueError(f"Plugin {plugin} is not a known FineTuningMethod")


def get_vectordb_by_fine_tuning_name(name: str) -> VectorDb:
    """
    Get the appropriate vector database for a plugin identified by name.

    This function looks up a plugin by name and returns the correct vector database
    for that plugin type.

    :param name: The name of the fine-tuning plugin
    :return: The appropriate VectorDb instance for the named plugin
    """
    plugin = context.plugin_manager.get_plugin(name)
    return get_vectordb(plugin)
