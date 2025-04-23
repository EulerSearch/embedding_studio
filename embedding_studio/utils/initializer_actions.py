import nltk
from apscheduler.schedulers.background import BackgroundScheduler

from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import CategoriesFineTuningMethod


def init_nltk():
    """
    Initialize NLTK by downloading the 'punkt' tokenizer.

    This function ensures that the punkt tokenizer, which is needed for sentence
    tokenization in NLTK, is downloaded and available for use.

    :return: None
    """
    nltk.download("punkt")


def init_plugin_manager():
    """
    Initialize the plugin manager by discovering and setting up enabled plugins.

    This function performs the following actions:
    1. Discovers plugins in the configured plugin directory
    2. For each plugin specified in INFERENCE_USED_PLUGINS:
       - Gets the inference client factory
       - Sets up vector database optimizations based on plugin type
       - Applies the optimizations to the vector database

    :return: None
    """
    context.plugin_manager.discover_plugins(directory=settings.ES_PLUGINS_PATH)
    for plugin_name in settings.INFERENCE_USED_PLUGINS:
        plugin = context.plugin_manager.get_plugin(plugin_name)
        plugin.get_inference_client_factory()

        has_optimization = False
        has_query_optimization = False
        for optimization in plugin.get_vectordb_optimizations():
            if issubclass(plugin.__class__, CategoriesFineTuningMethod):
                has_optimization = True
                context.vectordb.add_query_optimization(optimization)
            else:
                has_optimization = True
                context.vectordb.add_optimization(optimization)

        if has_optimization:
            context.vectordb.apply_optimizations()

        if has_query_optimization:
            context.vectordb.apply_query_optimizations()


def init_background_scheduler(task, seconds_interval: int):
    """
    Initialize a background scheduler to run a task at specified intervals.

    If a scheduler doesn't already exist in the application context, this function
    creates a new BackgroundScheduler. It then adds the specified task to run at
    the given interval and starts the scheduler.

    :param task: Callable function to be scheduled for periodic execution
    :param seconds_interval: Number of seconds between each execution of the task
    :return: None
    """
    if context.task_scheduler is None:
        context.task_scheduler = BackgroundScheduler()

    context.task_scheduler.add_job(task, "interval", seconds=seconds_interval)
    context.task_scheduler.start()
