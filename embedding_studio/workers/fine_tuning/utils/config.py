from typing import Dict

from pydantic import BaseModel


class RetryParams(BaseModel):
    max_attempts: int = 3
    wait_time_seconds: int = 3


class RetryConfig(BaseModel):
    default_params: RetryParams = RetryParams()
    _specific_retries: Dict[str, RetryParams] = dict()

    def __getitem__(self, item: str) -> RetryParams:
        return self._specific_retries.get(item, self.default_params)

    def __setitem__(self, key: str, value: RetryParams):
        self._specific_retries[key] = value
