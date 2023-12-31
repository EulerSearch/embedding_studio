import os

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.brokers.stub import StubBroker
from dramatiq_abort import Abortable, backends

from embedding_studio.core.config import settings

if os.getenv("ES_UNIT_TESTS") == "1":
    redis_broker = StubBroker()
    redis_broker.emit_after("process_boot")
else:
    redis_broker = RedisBroker(
        url=settings.REDIS_URL,
    )

abortable = Abortable(
    backend=backends.RedisBackend.from_url(settings.REDIS_URL)
)
redis_broker.add_middleware(abortable)

dramatiq.set_broker(redis_broker)
