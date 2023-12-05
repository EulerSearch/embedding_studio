import os

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.brokers.stub import StubBroker

from embedding_studio.core.config import settings

if os.getenv("ES_UNIT_TESTS") == "1":
    redis_broker = StubBroker()
    redis_broker.emit_after("process_boot")
else:
    redis_broker = RedisBroker(
        url=settings.REDIS_URL,
    )

dramatiq.set_broker(redis_broker)
