import nltk

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings


def init_nltk():
    nltk.download("punkt")


def init_plugin_manager():
    context.plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)
    for plugin_name in settings.INFERENCE_USED_PLUGINS:
        plugin = context.plugin_manager.get_plugin(plugin_name)
        plugin.get_inference_client_factory()
