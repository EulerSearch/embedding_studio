import json
import os
import sys
import uuid
from io import BytesIO

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from tqdm.auto import tqdm

# TODO: clean up sessions before running.

print(
    "Welcome to the EmbeddingStudio!\n"
    "In this demo, we will show how to emulate a clickstream and "
    "send it to the fine-tuning worker.\n\n"
    "First, we will load the click sessions from S3. "
    "Let's proceed with the necessary setup..."
)

s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

BUCKET_NAME = "embedding-studio-experiments"
CLICKSTREAM_INFO_KEY = (
    "remote-lanscapes/clickstream/"
    "f6816566-cac3-46ac-b5e4-0d5b76757c93/sessions.json"
)

try:
    response = s3_client.get_object(
        Bucket=BUCKET_NAME, Key=CLICKSTREAM_INFO_KEY
    )
    data = response["Body"].read()
    in_memory_file = BytesIO(data)
except Exception as e:
    print(f"Error downloading object: {e}")
    sys.exit(1)

clickstream = json.loads(in_memory_file.getvalue().decode("utf-8"))
print(
    f"Loading complete. Number of objects in clickstream: "
    f"{len(clickstream)}"
)

print("Uploading clickstream data to EmbeddingStudio...")
ES_URL = os.getenv("ES_URL", "http://embedding_studio:5000")
ES_CREATE_TASK_URL = f"{ES_URL}/api/v1/fine-tuning/task"
ES_CREATE_SESSION_URL = f"{ES_URL}/api/v1/clickstream/session"
ES_SESSION_PUSH_EVENTS_URL = f"{ES_URL}/api/v1/clickstream/session/events"

# Let's take only 20 sessions.
FIRST_N_SESSIONS = 20

for session_id, session in enumerate(tqdm(clickstream[:FIRST_N_SESSIONS])):
    session_id = str(session_id)
    search_results = []
    events = []
    for result_info in session["results"]:
        search_results.append(
            dict(
                object_id=result_info["item"]["file"],
                rank=result_info["rank"],
                meta=result_info["item"],
            )
        )
        if result_info["is_click"]:
            events.append(
                dict(
                    event_id=str(uuid.uuid4()),
                    object_id=result_info["item"]["file"],
                )
            )

    # And create a new search session.
    response = requests.post(
        ES_CREATE_SESSION_URL,
        json=dict(
            session_id=session_id,
            search_query=session["query"]["text"],
            search_meta=session["query"],
            search_results=search_results,
        ),
    )
    if events:
        reponse = requests.post(
            ES_SESSION_PUSH_EVENTS_URL,
            json=dict(session_id=session_id, events=events),
        )
print("Clickstream data successfully uploaded.")
print(
    "Now we can start the model fine-tuning. "
    "For this, we need to create a task..."
)
