import logging
import time

from requests.exceptions import ConnectionError, RequestException, Timeout

from embedding_studio.worker.utils.exceptions import (
    MaxAttemptsReachedException,
)

logger = logging.getLogger(__name__)


def retry_method(name: str = None):
    def decorator(func):
        """Decorator to run provided class method with attempts"""

        def wrapper(self, *args, **kwargs):
            func_name = name if name else func.__name__
            retry_params = self.retry_config[func_name]

            if (
                retry_params.max_attempts is None
                or retry_params.max_attempts <= 1
            ):
                return func(self, *args, **kwargs)

            attempts = 0
            exception = None
            while attempts < retry_params.max_attempts:
                try:
                    result = func(self, *args, **kwargs)
                    # If the function succeeds, return the result
                    return result
                except RequestException as e:
                    if (
                        hasattr(e, "response")
                        and e.response is not None
                        and 500 <= e.response.status_code < 600
                    ):
                        logger.error(
                            f"Server Error (5xx): {e.response.status_code}"
                        )
                        # Handle server error appropriately, e.g., retry, log, or raise a custom exception
                        exception = e
                    else:
                        logger.exception(f"Request Exception: {e}")
                        raise e

                except Timeout as e:
                    logger.error(f"Timeout: {e}")
                    exception = e

                except ConnectionError as e:
                    logger.error(f"Connection error: {e}")
                    exception = e

                except Exception as e:  # Handle other request exceptions
                    if (
                        hasattr(self, "attempt_exception_types")
                        and type(e) in self.attempt_exception_types
                    ) or (
                        hasattr(self, "is_retryable_error")
                        and self.is_retryable_error(e)
                    ):
                        logger.error(
                            f"Catch exception with type {type(e).__name__} that leads to new attempt"
                        )
                        exception = e
                    else:
                        raise

                if exception is not None:
                    logger.info(
                        f"Attempt {attempts + 1} failed with error: {exception}"
                    )
                    attempts += 1
                    time.sleep(retry_params.wait_time_seconds)

            raise MaxAttemptsReachedException(
                retry_params.max_attempts
            ) from exception

        return wrapper

    return decorator
