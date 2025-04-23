import logging
from typing import Any, Callable, Dict, Optional

from embedding_studio.core.config import settings
from embedding_studio.utils.retry import retry_method
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)

logger = logging.getLogger(__name__)


class ModelDownloader:
    def __init__(
        self,
        retry_config: Optional[Dict[str, Any]] = None,
    ):
        """
        A manager that handles downloading with retry logic.

        :param retry_config: Configuration for retry behavior.
        """
        self.retry_config = retry_config
        if self.retry_config is None:
            default_retry_params = RetryParams(
                max_attempts=settings.DEFAULT_MAX_ATTEMPTS,
                wait_time_seconds=settings.DEFAULT_WAIT_TIME_SECONDS,
            )

            self.retry_config = RetryConfig(
                default_params=default_retry_params
            )
            self.retry_config["download_model"] = RetryParams(
                max_attempts=settings.MODEL_DOWNLOAD_MAX_ATTPEMPTS,
                wait_time_seconds=settings.MODEL_DOWNLOAD_WAIT_TIME_SECONDS,
            )

    @retry_method(name="download_model")
    def download_model(
        self,
        model_name: str,
        download_fn: Callable[[str], Any],
    ) -> Any:
        """
        Download the model using the provided download function with retries.

        :param model_name: The name of the model to download.
        :param download_fn: A callable function to perform the download.
        :return: The downloaded model or tokenizer.
        :raises: MaxAttemptsReachedException if the download fails
                  after the maximum number of attempts.
        """
        logger.info(f"Attempting to download model: {model_name}")
        try:
            model = download_fn(model_name)
            logger.info(f"Successfully downloaded model: {model_name}")
            return model
        except Exception as e:
            logger.error(
                f"Failed to download model: {model_name} with error: {str(e)}"
            )
            raise
