import logging
import time
from functools import wraps

from requests.exceptions import ConnectionError, RequestException, Timeout

from embedding_studio.workers.fine_tuning.utils.exceptions import (
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


def retry_function(
    max_attempts: int = 3,
    wait_time_seconds: float = 2.0,
    attempt_exception_types: tuple = (
        RequestException,
        Timeout,
        ConnectionError,
    ),
    is_retryable_error=None,
):
    """
    Decorator to retry a function execution on failure.

    :param max_attempts: Maximum number of attempts before failing.
    :param wait_time_seconds: Time to wait between attempts.
    :param attempt_exception_types: Tuple of exception types that should trigger a retry.
    :param is_retryable_error: Optional function to determine if an exception is retryable.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            exception = None

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)  # Execute function
                except attempt_exception_types as e:
                    # Check if the error is retryable (server error or custom function)
                    if (
                        isinstance(e, RequestException)
                        and hasattr(e, "response")
                        and e.response is not None
                        and 500 <= e.response.status_code < 600
                    ):
                        logger.error(
                            f"Server Error (5xx): {e.response.status_code}"
                        )
                        exception = e
                    elif is_retryable_error and is_retryable_error(e):
                        logger.error(
                            f"Custom retryable exception: {type(e).__name__}"
                        )
                        exception = e
                    else:
                        logger.exception(
                            f"Non-retryable exception: {type(e).__name__}"
                        )
                        raise  # Stop retrying if the exception is not explicitly retryable

                except Exception as e:
                    logger.exception(
                        f"Unexpected exception: {type(e).__name__}"
                    )
                    raise  # Do not retry unexpected exceptions

                logger.info(
                    f"Attempt {attempts + 1} failed with error: {exception}"
                )
                attempts += 1
                time.sleep(wait_time_seconds)

            raise MaxAttemptsReachedException(max_attempts) from exception

        return wrapper

    return decorator
