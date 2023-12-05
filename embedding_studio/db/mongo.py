import os

from pymongo import MongoClient

from embedding_studio.core.config import settings

if os.getenv("ES_UNIT_TESTS") == "1":
    import mongomock

    mongo_client = mongomock.MongoClient(settings.MONGO_URL)
    database = mongo_client[settings.MONGO_DB_NAME]
else:
    mongo_client = MongoClient(settings.MONGO_URL)
    database = mongo_client[settings.MONGO_DB_NAME]
