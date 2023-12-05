import pytest
from dramatiq import Worker
from fastapi.testclient import TestClient

from embedding_studio.db.mongo import database
from embedding_studio.db.redis import redis_broker
from embedding_studio.main import app

pytest_plugins = ["tests.plugins.env_vars"]


@pytest.fixture()
def stub_broker():
    redis_broker.flush_all()
    return redis_broker


@pytest.fixture()
def stub_worker():
    worker = Worker(redis_broker, worker_timeout=100)
    worker.start()
    yield worker
    worker.stop()


@pytest.fixture(name="client")
def client_fixture():
    client = TestClient(app=app)
    yield client
    database["fine_tuning"].drop()
