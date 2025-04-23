import os

from pymongo import MongoClient

from embedding_studio.core.config import settings

MONGODB_UUID_REPRESENTATION = "standard"

if os.getenv("ES_UNIT_TESTS") == "1":
    import mongomock

    def _mongo_env_client(*args, **kwargs):
        return mongomock.MongoClient(*args, **kwargs)

else:

    def _mongo_env_client(*args, **kwargs):
        return MongoClient(
            directConnection=True, maxPoolSize=None, *args, **kwargs
        )


finetuning_mongo_client = _mongo_env_client(
    settings.FINETUNING_MONGO_URL,
    uuidRepresentation=MONGODB_UUID_REPRESENTATION,
    tz_aware=True,
)
sessions_for_improvement_mongo_client = _mongo_env_client(
    settings.SESSIONS_FOR_IMPROVEMENT_MONGO_URL,
    uuidRepresentation=MONGODB_UUID_REPRESENTATION,
    tz_aware=True,
)
inference_deployment_mongo_client = _mongo_env_client(
    settings.INFERENCE_DEPLOYMENT_MONGO_URL,
    uuidRepresentation=MONGODB_UUID_REPRESENTATION,
    tz_aware=True,
)
upsertion_mongo_client = _mongo_env_client(
    settings.UPSERTION_MONGO_URL,
    uuidRepresentation=MONGODB_UUID_REPRESENTATION,
    tz_aware=True,
)
clickstream_mongo_client = _mongo_env_client(
    settings.CLICKSTREAM_MONGO_URL,
    uuidRepresentation=MONGODB_UUID_REPRESENTATION,
    tz_aware=True,
)
embeddings_mongo_client = _mongo_env_client(
    settings.EMBEDDINGS_MONGO_URL,
    uuidRepresentation=MONGODB_UUID_REPRESENTATION,
    tz_aware=True,
)

finetuning_mongo_database = finetuning_mongo_client[
    settings.FINETUNING_MONGO_DB_NAME
]
sessions_for_improvement_mongo_database = (
    sessions_for_improvement_mongo_client[
        settings.SESSIONS_FOR_IMPROVEMENT_MONGO_DB_NAME
    ]
)
inference_deployment_mongo_database = inference_deployment_mongo_client[
    settings.INFERENCE_DEPLOYMENT_MONGO_DB_NAME
]
upsertion_mongo_database = upsertion_mongo_client[
    settings.UPSERTION_MONGO_DB_NAME
]
clickstream_mongo_database = clickstream_mongo_client[
    settings.CLICKSTREAM_MONGO_DB_NAME
]
embeddings_mongo_database = clickstream_mongo_client[
    settings.EMBEDDINGS_MONGO_DB_NAME
]
