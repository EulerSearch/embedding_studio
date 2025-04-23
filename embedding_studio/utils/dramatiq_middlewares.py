from typing import Any, Callable, List

from dramatiq import Middleware


class ActionsOnStartMiddleware(Middleware):
    """
    Middleware that executes specified actions when a Dramatiq worker starts.

    This middleware hooks into the Dramatiq worker lifecycle to run a list of
    callable actions after the worker has booted. This can be useful for
    initialization tasks like setting up connections, loading resources, or
    performing other setup operations.

    :param actions: A list of callables to execute after worker boot
    :return: A new ActionsOnStartMiddleware instance
    """

    def __init__(self, actions: List[Callable[[], Any]]):
        self.actions = actions

    def after_worker_boot(self, broker, worker):
        """
        Execute all registered actions after the worker has booted.

        This method is called by Dramatiq after a worker has been fully initialized.
        It calls the parent class method first, then executes each action in the
        order they were provided.

        :param broker: The broker instance the worker is connected to
        :param worker: The worker instance that has booted
        """
        super().after_worker_boot(broker, worker)
        for action in self.actions:
            action()
