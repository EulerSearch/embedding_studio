import logging
import uuid
from typing import Optional

import requests
from demo.utils.constants import DEFAULT_FINE_TUNING_METHOD_NAME

# TODO: Implement separate Python API Client

logger = logging.getLogger(__name__)


def release_batch(connection_url: str) -> Optional[str]:
    """We categorize sessions into batches. A 'batch' refers to new clickstream sessions that have not been utilized for fine-tuning.
    When initiating a new fine-tuning iteration, it's necessary to release the current batch.
    This allows new sessions to be assembled into the next batch, while the existing batch is being used.

    :param connection_url: EmbeddingStudio connection URL
    :return: released batch_id
    """
    release_id = str(uuid.uuid4())
    resp = requests.post(
        f"{connection_url}/api/v1/clickstream/internal/batch/release",
        json={"release_id": release_id},
    )

    if resp.status_code != 200:
        logger.error(
            f"Unable to release batch: {resp.status_code} status code, please check your logs"
        )

    else:
        batch_id = resp.json()["batch_id"]
        logger.info(f"Batch with ID {batch_id} was successfully released")
        return batch_id


def create_session_and_push_events(
    connection_url: str, session_id: str, session_info: dict
):
    search_results = []
    events = []
    for result_info in session_info["results"]:
        search_results.append(
            {
                "object_id": result_info["item"]["file"],
                "rank": result_info["rank"],
                "meta": result_info["item"],
            }
        )
        if result_info["is_click"]:
            # We add an event
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "object_id": result_info["item"]["file"],
                }
            )

    resp = requests.post(
        f"{connection_url}/api/v1/clickstream/session",
        json={
            "session_id": session_id,
            "search_query": session_info["query"]["dict"],
            "search_meta": session_info["query"],
            "search_results": search_results,
        },
    )

    if resp.status_code != 200:
        logger.error(
            f"Unable to create a session: {resp.status_code} / {resp.json()}"
        )

    elif len(events) > 0:
        events_resp = requests.post(
            f"{connection_url}/api/v1/clickstream/session/events",
            json={"session_id": session_id, "events": events},
        )

        if events_resp.status_code != 200:
            logger.error(
                f"Unable to push events: {events_resp.status_code} / {events_resp.json()}"
            )


def start_fine_tuning(
    connection_url: str, mlflow_url: str, batch_id: str
) -> Optional[str]:
    """Start fine-tuning task and wait until it's finished

    :param connection_url: EmbeddingStudio connection URL
    :param mlflow_url: MLFlow connection URL
    :param batch_id: released batch ID
    :return:
    """
    logger.info(
        f"Start fine-tuning task for the released batch with {batch_id} ID with default fine-tuning method"
    )
    resp = requests.post(
        f"{connection_url}/api/v1/fine-tuning/task",
        json={
            "fine_tuning_method": DEFAULT_FINE_TUNING_METHOD_NAME,
            "batch_id": batch_id,
        },
    )

    if resp.status_code != 200:
        logger.error(
            f"Unable to start fine-tuning task: {resp.status_code} status code, please check your logs"
        )
        return None

    else:
        task_id = resp.json()["id"]
        logger.info(
            f"Fine-tuning task with {task_id} ID was successfully started"
        )

        return task_id


def get_task_status(connection_url: str, task_id: str) -> Optional[str]:
    """Get status of fine-tuning task.

    :param connection_url:  EmbeddingStudio connection URL
    :param task_id: fine-tuning task ID
    :return:
    """
    status_resp = requests.get(
        f"{connection_url}/api/v1/fine-tuning", params={"id": task_id}
    )
    if status_resp.status_code != 200:
        logger.error(
            f"Unable to get fine-tuning task {task_id} status: {status_resp.status_code} status code, please check your logs"
        )
        return None
    else:
        return status_resp.json()["status"]
