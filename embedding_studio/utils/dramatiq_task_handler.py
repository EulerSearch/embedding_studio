import logging
from typing import Any, Callable, Optional

from pymongo.errors import PyMongoError

from embedding_studio.data_access.mongo.crud_base import CRUDBase

logger = logging.getLogger(__name__)


def update_task_with_retry(
    task: Any, broker_id: str, task_manager: CRUDBase
) -> bool:
    try:
        task.broker_id = broker_id
        # Assume that the task has an id attribute that indicates the task type
        task_manager.update(obj=task)
        logger.info(f"Updated task {task.id} with broker_id {broker_id}")
        return True
    except PyMongoError as e:
        logger.error(
            f"Failed to update task {task.id} with broker_id {broker_id}. Error: {str(e)}"
        )
        raise


def create_and_send_task(
    worker: Callable, task: Any, task_crud: CRUDBase
) -> Optional[Any]:
    try:
        # Create a message, but don't send it yet
        message = worker.message(str(task.id))

        # Try to update the task with the broker_id
        if update_task_with_retry(task, message.message_id, task_crud):
            # If the update was successful, send the message
            worker.send(str(task.id))
            logger.debug(
                f"Worker message sent: {message.message_id} for task {task.id}"
            )
            return task
        else:
            logger.error(f"Failed to update task {task.id} with broker_id")
            return None
    except Exception as e:
        logger.exception(
            f"Error during task creation and sending for task {task.id}: {str(e)}"
        )
        return None
