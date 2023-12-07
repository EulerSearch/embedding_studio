import os
import secrets
from typing import List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8'
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


settings = Settings()
