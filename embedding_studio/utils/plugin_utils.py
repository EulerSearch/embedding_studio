from embedding_studio.context.app_context import context
from embedding_studio.core.plugin import (
    CategoriesFineTuningMethod,
    FineTuningMethod,
)
from embedding_studio.vectordb.vectordb import VectorDb


def is_basic_plugin(plugin: FineTuningMethod) -> bool:
    return issubclass(plugin, FineTuningMethod) and not issubclass(
        plugin, CategoriesFineTuningMethod
    )


def is_categories_plugin(plugin: FineTuningMethod) -> bool:
    return issubclass(plugin, CategoriesFineTuningMethod)


def get_vectordb(plugin: FineTuningMethod) -> VectorDb:
    if is_basic_plugin(plugin):
        return context.vectordb

    elif is_categories_plugin(plugin):
        return context.vectordb_categories

    else:
        raise ValueError(f"Plugin {plugin} is not a known FineTuningMethod")


def get_vectordb_by_fine_tuning_name(name: str) -> VectorDb:
    plugin = context.plugin_manager.get_plugin(name)
    return get_vectordb(plugin)
