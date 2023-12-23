import os

import requests

ES_URL = os.environ.get("ES_URL", "http://embedding_studio:5000")
ES_TASK_URL = f"{ES_URL}/api/v1/fine-tuning/task"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://mlflow:5000"
)

response = requests.post(
    ES_TASK_URL,
    json=dict(
        fine_tuning_method="Default Fine Tuning Method",
    ),
)
task_id = response.json()["id"]
print(f"Task has been created, task_id={task_id}")
print(
    f"You can check the progress and results of the training in "
    f"MlFlow: {MLFLOW_TRACKING_URI}"
)
