import logging
from typing import Any, Callable, Optional

from pymongo.errors import PyMongoError

from embedding_studio.data_access.mongo.crud_base import CRUDBase

logger = logging.getLogger(__name__)


def update_task_with_retry(
    task: Any, broker_id: str, task_manager: CRUDBase
) -> bool:
    """
    Update a task with a broker ID and handle potential MongoDB errors.

    This function attempts to update a task in the database with the associated
    broker ID from the message queue. If the operation fails due to a MongoDB error,
    it logs the error and re-raises the exception.

    :param task: The task object to update
    :param broker_id: The broker ID to associate with the task
    :param task_manager: CRUD manager for task operations
    :return: True if the update was successful
    """
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
    """
    Create a message for a task, update the task with the broker ID, and send it.

    This function handles the complete process of:
    1. Creating a Dramatiq message for the task
    2. Updating the task in the database with the message's broker ID
    3. Sending the task to the message queue if the database update succeeds

    Any exceptions during this process are caught, logged, and will result in a None return.

    :param worker: The Dramatiq actor that will process the task
    :param task: The task object to be sent to the message queue
    :param task_crud: CRUD manager for task operations
    :return: The updated task object if successful, None otherwise
    """
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
