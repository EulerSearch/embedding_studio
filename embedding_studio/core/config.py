import os
import secrets
from typing import Any, List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8"
    )

    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    # MongoDB
    MONGO_HOST: str = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT: int = os.getenv("MONGO_PORT", 27017)
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "emdegginstudio")
    MONGO_USERNAME: str = os.getenv("MONGO_USERNAME", "root")
    MONGO_PASSWORD: str = os.getenv("MONGO_PASSWORD", "mongopassword")
    MONGO_URL: str = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@"
        f"{MONGO_HOST}:{MONGO_PORT}"
    )

    # Redis (broker for dramatiq)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = os.getenv("REDIS_PORT", 6379)
    REDIS_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

    # S3
    S3_HOST: str = os.getenv("S3_HOST", "localhost")
    S3_PORT: int = os.getenv("S3_PORT", 9000)
    S3_ACCESS_KEY_ID: str = os.getenv("S3_ACCESS_KEY_ID", "root")
    S3_SECRET_ACCESS_KEY: str = os.getenv(
        "S3_SECRET_ACCESS_KEY", "miniopassword"
    )
    S3_BUCKET: str = os.getenv("S3_BUCKET", "embeddingstudio_bucket")

    # Retry strategy
    DEFAULT_MAX_ATTEMPTS: int = os.getenv("DEFAULT_MAX_ATTEMPTS", 3)
    DEFAULT_WAIT_TIME_SECONDS: float = os.getenv("DEFAULT_WAIT_TIME_SECONDS", 3.0)

    # S3
    S3_READ_CREDENTIALS_ATTEMPTS: int = os.getenv(
        "S3_READ_CREDENTIALS_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    S3_READ_WAIT_TIME_SECONDS: float = os.getenv(
        "S3_READ_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    S3_DOWNLOAD_DATA_ATTEMPTS: int = os.getenv(
        "S3_DOWNLOAD_DATA_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    S3_DOWNLOAD_DATA_WAIT_TIME_SECONDS: float = os.getenv(
        "S3_DOWNLOAD_DATA_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    # Experiments manager
    MLFLOW_LOG_METRIC_ATTEMPTS: int = os.getenv(
        "MLFLOW_LOG_METRIC_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_LOG_METRIC_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_LOG_METRIC_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_LOG_PARAM_ATTEMPTS: int = os.getenv(
        "MLFLOW_LOG_PARAM_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_LOG_PARAM_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_LOG_PARAM_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_LOG_MODEL_ATTEMPTS: int = os.getenv(
        "MLFLOW_LOG_MODEL_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_LOG_MODEL_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_LOG_MODEL_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_LOAD_MODEL_ATTEMPTS: int = os.getenv(
        "MLFLOW_LOAD_MODEL_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_LOAD_MODEL_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_LOAD_MODEL_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_DELETE_MODEL_ATTEMPTS: int = os.getenv(
        "MLFLOW_DELETE_MODEL_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_DELETE_MODEL_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_DELETE_MODEL_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_SEARCH_RUNS_ATTEMPTS: int = os.getenv(
        "MLFLOW_SEARCH_RUNS_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_SEARCH_RUNS_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_SEARCH_RUNS_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_END_RUN_ATTEMPTS: int = os.getenv(
        "MLFLOW_END_RUN_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_END_RUN_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_END_RUN_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_GET_RUN_ATTEMPTS: int = os.getenv(
        "MLFLOW_GET_RUN_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_GET_RUN_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_GET_RUN_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_SEARCH_EXPERIMENTS_ATTEMPTS: int = os.getenv(
        "MLFLOW_SEARCH_EXPERIMENTS_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_SEARCH_EXPERIMENTS_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_SEARCH_EXPERIMENTS_WAIT_TIME_SECONDS",
        DEFAULT_WAIT_TIME_SECONDS,
    )

    MLFLOW_DELETE_EXPERIMENT_ATTEMPTS: int = os.getenv(
        "MLFLOW_DELETE_EXPERIMENT_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_DELETE_EXPERIMENT_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_DELETE_EXPERIMENT_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_CREATE_EXPERIMENT_ATTEMPTS: int = os.getenv(
        "MLFLOW_CREATE_EXPERIMENT_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_CREATE_EXPERIMENT_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_CREATE_EXPERIMENT_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )

    MLFLOW_GET_EXPERIMENT_ATTEMPTS: int = os.getenv(
        "MLFLOW_GET_EXPERIMENT_ATTEMPTS", DEFAULT_MAX_ATTEMPTS
    )
    MLFLOW_GET_EXPERIMENT_WAIT_TIME_SECONDS: float = os.getenv(
        "MLFLOW_GET_EXPERIMENT_WAIT_TIME_SECONDS", DEFAULT_WAIT_TIME_SECONDS
    )


settings = Settings()
