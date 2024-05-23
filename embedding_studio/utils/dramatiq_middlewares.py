from typing import Any, Callable, List

from dramatiq import Middleware


class ActionsOnStartMiddleware(Middleware):
    """
    Middleware to handle worker start events.
    """

    def __init__(self, actions: List[Callable[[], Any]]):
        self.actions = actions

    def after_worker_boot(self, broker, worker):
        super().after_worker_boot(broker, worker)
        for action in self.actions:
            action()
