from typing import Dict, List, NamedTuple, Union

import pytest
from fastapi.testclient import TestClient
from freezegun import freeze_time


class Params(NamedTuple):
    requests: List[Dict[str, str]]
    responses: List[Dict[str, Union[str, int]]]
    preload_data: List[Dict[str, str]] = [
        {
            "method": "POST",
            "url": "/api/v1/fine-tuning/task",
            "json": {
                "fine_tuning_method": "Test Method 1",
            },
        },
        {
            "method": "POST",
            "url": "/api/v1/fine-tuning/task",
            "json": {
                "fine_tuning_method": "Test Method 2",
                "metadata": {"some": "testmetadata"},
            },
        },
    ]


@freeze_time("2023-12-15T11:15:00")
def test_fine_tuning_base(client: TestClient):
    response = client.post(
        url="/api/v1/fine-tuning/task",
        json={
            "fine_tuning_method": "Test Method",
        },
    )
    assert response.status_code == 200
    task_id = response.json()["id"]

    response = client.get(url=f"/api/v1/fine-tuning/task/{task_id}")
    assert response.status_code == 200
    assert response.json() == {
        "fine_tuning_method": "Test Method",
        "id": task_id,
        "status": "pending",
        "created_at": "2023-12-15T11:15:00",
        "updated_at": "2023-12-15T11:15:00",
    }

    response = client.get(url="/api/v1/fine-tuning/task")
    assert response.status_code == 200
    assert response.json() == [
        {
            "fine_tuning_method": "Test Method",
            "id": task_id,
            "status": "pending",
            "created_at": "2023-12-15T11:15:00",
            "updated_at": "2023-12-15T11:15:00",
        }
    ]


# TODO: need to add more tests
@freeze_time("2023-12-15T11:15:00")
@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            Params(
                preload_data=[],
                requests=[
                    {
                        "method": "POST",
                        "url": "/api/v1/fine-tuning/task",
                        "json": {
                            "fine_tuning_method": "Test Method",
                        },
                    }
                ],
                responses=[
                    {
                        "status_code": 200,
                        "json": {
                            "fine_tuning_method": "Test Method",
                            "status": "pending",
                            "created_at": "2023-12-15T11:15:00",
                            "updated_at": "2023-12-15T11:15:00",
                        },
                    }
                ],
            ),
            id="create task",
        ),
        pytest.param(
            Params(
                requests=[
                    {
                        "method": "GET",
                        "url": "/api/v1/fine-tuning/task",
                    },
                    {
                        "method": "GET",
                        "url": "/api/v1/fine-tuning/task/",
                        "use_index": 0,
                    },
                    {
                        "method": "GET",
                        "url": "/api/v1/fine-tuning/task/",
                        "use_index": 1,
                    },
                ],
                responses=[
                    {
                        "status_code": 200,
                        "json": [
                            {
                                "fine_tuning_method": "Test Method 1",
                                "status": "pending",
                                "created_at": "2023-12-15T11:15:00",
                                "updated_at": "2023-12-15T11:15:00",
                            },
                            {
                                "fine_tuning_method": "Test Method 2",
                                "metadata": {"some": "testmetadata"},
                                "status": "pending",
                                "created_at": "2023-12-15T11:15:00",
                                "updated_at": "2023-12-15T11:15:00",
                            },
                        ],
                    },
                    {
                        "status_code": 200,
                        "json": {
                            "fine_tuning_method": "Test Method 1",
                            "status": "pending",
                            "created_at": "2023-12-15T11:15:00",
                            "updated_at": "2023-12-15T11:15:00",
                        },
                    },
                    {
                        "status_code": 200,
                        "json": {
                            "fine_tuning_method": "Test Method 2",
                            "metadata": {"some": "testmetadata"},
                            "status": "pending",
                            "created_at": "2023-12-15T11:15:00",
                            "updated_at": "2023-12-15T11:15:00",
                        },
                    },
                ],
            ),
            id="get task",
        ),
        pytest.param(
            Params(
                requests=[
                    {
                        "method": "GET",
                        "url": "/api/v1/fine-tuning/task?status=pending",
                    },
                ],
                responses=[
                    {
                        "status_code": 200,
                        "json": [
                            {
                                "fine_tuning_method": "Test Method 1",
                                "status": "pending",
                                "created_at": "2023-12-15T11:15:00",
                                "updated_at": "2023-12-15T11:15:00",
                            },
                            {
                                "fine_tuning_method": "Test Method 2",
                                "metadata": {"some": "testmetadata"},
                                "status": "pending",
                                "created_at": "2023-12-15T11:15:00",
                                "updated_at": "2023-12-15T11:15:00",
                            },
                        ],
                    },
                ],
            ),
            id="get tasks by status",
        ),
    ],
)
def test_fine_tuning_all(client: TestClient, params: Params):
    assert len(params.requests) == len(
        params.responses
    ), "the number of requests and responses does not match"

    tasks_id: List[str] = []
    for pre_data in params.preload_data:
        response = client.request(
            method=pre_data.get("method"),
            url=pre_data.get("url"),
            json=pre_data.get("json"),
        )
        tasks_id.append(response.json().get("id"))

    for i, request in enumerate(params.requests):
        if "use_index" in request:
            url = request.get("url") + tasks_id[request.get("use_index")]
        else:
            url = request.get("url")

        response = client.request(
            method=request.get("method"), url=url, json=request.get("json")
        )
        assert response.status_code == params.responses[i].get("status_code")

        response_json = response.json()
        if not params.preload_data:
            tasks_id.append(response_json.get("id"))

        if isinstance(response_json, list):
            for res in response_json:
                res.pop("id")
        elif isinstance(response_json, dict):
            response_json.pop("id")

        assert response_json == params.responses[i].get("json")
