from sqlalchemy import create_engine

from embedding_studio.core.config import settings

pg_database = create_engine(
    settings.POSTGRES_DB_URI,
    pool_pre_ping=True,
    connect_args={
        "sslmode": "disable",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
        "prepare_threshold": 1,
    },
)
