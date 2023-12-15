import dataclasses

from embedding_studio.data_access.clickstream import ClickstreamDao
from embedding_studio.data_access.mongo.clickstream import MongoClickstreamDao
from embedding_studio.db import mongo


@dataclasses.dataclass
class AppContext:
    clickstream_dao: ClickstreamDao


context = AppContext(
    clickstream_dao=MongoClickstreamDao(
        mongo_database=mongo.clckstream_mongo_database
    )
)
