import logging
import time

import dramatiq

from embedding_studio.crud.fine_tuning import fine_tuning_task
from embedding_studio.schemas.fine_tuning import FineTuningStatus

logger = logging.getLogger(__name__)


class FineTuningException(Exception):
    pass


@dramatiq.actor(queue_name="fine_tuning_worker", max_retries=3)
def fine_tuning_worker(task_id: str):
    logger.info(f"start fine_tuning_worker: task_id={task_id}")
    time.sleep(5)  # only for test, need to delete
    # load datasets
    task = fine_tuning_task.get(id=task_id)
    if not task:
        raise FineTuningException(f"Task with ID `{task_id}` not found")
    task.status = FineTuningStatus.processing
    fine_tuning_task.update(obj=task)
