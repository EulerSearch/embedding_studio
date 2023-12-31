import os
import secrets
from typing import List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


# noinspection SpellCheckingInspection
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
    FINETUNING_MONGO_HOST: str = os.getenv("FINETUNING_MONGO_HOST", "mongo")
    FINETUNING_MONGO_PORT: int = os.getenv("FINETUNING_MONGO_PORT", 27017)
    FINETUNING_MONGO_DB_NAME: str = os.getenv(
        "FINETUNING_MONGO_DB_NAME", "embedding_studio"
    )

    FINETUNING_MONGO_USERNAME: str = os.getenv(
        "FINETUNING_MONGO_USERNAME", "root"
    )
    FINETUNING_MONGO_PASSWORD: str = os.getenv(
        "FINETUNING_MONGO_PASSWORD", "mongopassword"
    )
    FINETUNING_MONGO_URL: str = (
        f"mongodb://{FINETUNING_MONGO_USERNAME}:{FINETUNING_MONGO_PASSWORD}@"
        f"{FINETUNING_MONGO_HOST}:{FINETUNING_MONGO_PORT}"
    )

    CLICKSTREAM_MONGO_HOST: str = os.getenv("CLICKSTREAM_MONGO_HOST", "mongo")
    CLICKSTREAM_MONGO_PORT: int = os.getenv("CLICKSTREAM_MONGO_PORT", 27017)
    CLICKSTREAM_MONGO_DB_NAME: str = os.getenv(
        "CLICKSTREAM_MONGO_DB_NAME", "embedding_studio"
    )

    CLICKSTREAM_MONGO_USERNAME: str = os.getenv(
        "CLICKSTREAM_MONGO_USERNAME", "root"
    )
    CLICKSTREAM_MONGO_PASSWORD: str = os.getenv(
        "CLICKSTREAM_MONGO_PASSWORD", "mongopassword"
    )
    CLICKSTREAM_MONGO_URL: str = (
        f"mongodb://{CLICKSTREAM_MONGO_USERNAME}:{CLICKSTREAM_MONGO_PASSWORD}@"
        f"{CLICKSTREAM_MONGO_HOST}:{CLICKSTREAM_MONGO_PORT}"
    )

    # Redis (broker for dramatiq)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = os.getenv("REDIS_PORT", 6379)
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "redispassword")
    REDIS_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

    # minio
    MINIO_HOST: str = os.getenv("MINIO_HOST", "localhost")
    MINIO_PORT: int = os.getenv("MINIO_PORT", 9000)
    MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "root")
    MINIO_ROOT_PASSWORD: str = os.getenv(
        "MINIO_ROOT_PASSWORD", "miniopassword"
    )
    MINIO_DEFAULT_BUCKETS: str = os.getenv(
        "MINIO_DEFAULT_BUCKETS", "embeddingstudio"
    )
    MINIO_ACCESS_KEY: str = os.getenv(
        "MINIO_ACCESS_KEY", "mtGNiEvoTL6C0EXAMPLE"
    )
    MINIO_SECRET_KEY: str = os.getenv(
        "MINIO_SECRET_KEY", "HY5JserXAaWmphNyCpQPEXAMPLEKEYEXAMPLEKEY"
    )

    # mysql (for mlflow)
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = os.getenv("MYSQL_PORT", 3306)
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "mlflow")
    MYSQL_USER: str = os.getenv("MYSQL_USER", "mlflow_user")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "Baxp3O5rUvpIxiD77BfZ")
    MYSQL_ROOT_PASSWORD: str = os.getenv(
        "MYSQL_ROOT_PASSWORD", "PrK5qmPTDsm2IYKvHVG8"
    )

    # mlflow
    MLFLOW_HOST: str = os.getenv("MLFLOW_HOST", "localhost")
    MLFLOW_PORT: int = os.getenv("MLFLOW_PORT", 5001)
    MLFLOW_TRACKING_URI: str = f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"

    # Plugins
    ES_PLUGINS_PATH: str = os.getenv("ES_PLUGINS_PATH", "plugins")

    # Fine-tuning worker
    FINE_TUNING_WORKER_MAX_RETRIES: int = os.getenv(
        "FINE_TUNING_WORKER_MAX_RETRIES", 3
    )
    FINE_TUNING_WORKER_TIME_LIMIT: int = os.getenv(
        "FINE_TUNING_WORKER_TIME_LIMIT", 18000000
    )

    # Retry strategy
    DEFAULT_MAX_ATTEMPTS: int = os.getenv("DEFAULT_MAX_ATTEMPTS", 3)
    DEFAULT_WAIT_TIME_SECONDS: float = os.getenv(
        "DEFAULT_WAIT_TIME_SECONDS", 3.0
    )

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

    # Clickstream
    CLICKSTREAM_TIME_MAX_DELTA_MINUS_SEC: int = os.getenv(
        "CLICKSTREAM_TIME_MAX_DELTA_MINUS_SEC", 12 * 60 * 60
    )
    CLICKSTREAM_TIME_MAX_DELTA_PLUS_SEC: int = os.getenv(
        "CLICKSTREAM_TIME_MAX_DELTA_PLUS_SEC", 5 * 60
    )


settings = Settings()
