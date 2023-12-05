from typing import Dict, List, NamedTuple, Union

import pytest
from fastapi.testclient import TestClient


class Params(NamedTuple):
    requests: List[Dict[str, str]]
    responses: List[Dict[str, Union[str, int]]]
    preload_data: List[Dict[str, str]] = [
        {
            "method": "POST",
            "url": "/api/v1/fine-tuning/create",
            "json": {
                "start_at": "2022-11-29T10:00:00",
                "end_at": "2022-11-30T10:00:00",
                "metadata": {},
            },
        },
        {
            "method": "POST",
            "url": "/api/v1/fine-tuning/create",
            "json": {
                "start_at": "2023-01-01T00:00:00",
                "end_at": "2023-12-30T23:59:59",
                "metadata": {},
            },
        },
    ]


def test_fine_tuning_base(client: TestClient):
    response = client.post(
        url="/api/v1/fine-tuning/create",
        json={
            "start_at": "2022-11-29T10:00:00",
            "end_at": "2022-11-30T10:00:00",
            "metadata": {},
        },
    )
    assert response.status_code == 201
    task_id = response.json()["id"]

    response = client.get(url=f"/api/v1/fine-tuning/get/{task_id}")
    assert response.status_code == 200
    assert response.json() == {
        "start_at": "2022-11-29T10:00:00",
        "end_at": "2022-11-30T10:00:00",
        "metadata": {},
        "id": task_id,
        "status": "pending",
    }

    response = client.get(url="/api/v1/fine-tuning/get")
    assert response.status_code == 200
    assert response.json() == [
        {
            "start_at": "2022-11-29T10:00:00",
            "end_at": "2022-11-30T10:00:00",
            "metadata": {},
            "id": task_id,
            "status": "pending",
        }
    ]


# TODO: need to add more tests
@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            Params(
                preload_data=[],
                requests=[
                    {
                        "method": "POST",
                        "url": "/api/v1/fine-tuning/create",
                        "json": {
                            "start_at": "2022-11-29T10:00:00",
                            "end_at": "2022-11-30T10:00:00",
                            "metadata": {},
                        },
                    }
                ],
                responses=[
                    {
                        "status_code": 201,
                        "json": {
                            "start_at": "2022-11-29T10:00:00",
                            "end_at": "2022-11-30T10:00:00",
                            "metadata": {},
                            "status": "pending",
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
                        "url": "/api/v1/fine-tuning/get",
                    },
                    {
                        "method": "GET",
                        "url": "/api/v1/fine-tuning/get/",
                        "use_index": 0,
                    },
                    {
                        "method": "GET",
                        "url": "/api/v1/fine-tuning/get/",
                        "use_index": 1,
                    },
                ],
                responses=[
                    {
                        "status_code": 200,
                        "json": [
                            {
                                "start_at": "2022-11-29T10:00:00",
                                "end_at": "2022-11-30T10:00:00",
                                "metadata": {},
                                "status": "pending",
                            },
                            {
                                "start_at": "2023-01-01T00:00:00",
                                "end_at": "2023-12-30T23:59:59",
                                "metadata": {},
                                "status": "pending",
                            },
                        ],
                    },
                    {
                        "status_code": 200,
                        "json": {
                            "start_at": "2022-11-29T10:00:00",
                            "end_at": "2022-11-30T10:00:00",
                            "metadata": {},
                            "status": "pending",
                        },
                    },
                    {
                        "status_code": 200,
                        "json": {
                            "start_at": "2023-01-01T00:00:00",
                            "end_at": "2023-12-30T23:59:59",
                            "metadata": {},
                            "status": "pending",
                        },
                    },
                ],
            ),
            id="get task",
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
