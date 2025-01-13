import nltk
from apscheduler.schedulers.background import BackgroundScheduler

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings


def init_nltk():
    nltk.download("punkt")


def init_plugin_manager():
    context.plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)
    for plugin_name in settings.INFERENCE_USED_PLUGINS:
        plugin = context.plugin_manager.get_plugin(plugin_name)
        plugin.get_inference_client_factory()


def init_background_scheduler(task, seconds_interval: int):
    if context.task_scheduler is None:
        context.task_scheduler = BackgroundScheduler()

    context.task_scheduler.add_job(task, "interval", seconds=seconds_interval)
    context.task_scheduler.start()
