import argparse
import json
import logging
import time
from typing import Optional

import boto3  # as we put data in S3, we need to use this package
from botocore import UNSIGNED
from botocore.client import Config
from demo.utils.api_utils import (
    create_session_and_push_events,
    get_task_status,
    release_batch,
    start_fine_tuning,
)
from demo.utils.aws_utils import download_s3_object_to_memory
from demo.utils.mlflow_utils import get_mlflow_results_url
from tqdm.auto import tqdm

DESCRIPTION = """
# Introduction

EmbeddingStudio is the open-source framework, that allows you to transform a joint "Embedding Model + Vector DB" into
a full-cycle search engine: collect clickstream -> improve search experience -> adapt embedding model and repeat out-of-the-box.

This is the demo. The main target is to help you to test out EmbeddingStudio locally. And then make a decision to use it or not.

In this demo we help you step-by-step to run and test EmbeddingStudio. This is the part of the demo,
when we emulate **user clickstream** and run **fine-tuning procedure**. We expect that you:

* Checked all system requirements
* Installed Docker and docker-compose
* Have already run all system using provided docker-compose
* Initialized experiments tracker

## What do we mean by the word "Emulate"?

It's all simple:
1. We picked the easiest domain and the easiest dataset ([Remote landscapes](https://huggingface.co/datasets/EmbeddingStudio/merged_remote_landscapes_v1)),
so we definitely can show positive results in the demo;
2. We generated related dict queries using GPT3.5;
3. And for each generated dict query we emulated search sessions and user clicks (with some probability of a mistake);
4. All data we put into public to read AWS S3 bucket;


## Dataset

As I mentioned before for the demo we use **the easiest domain and dataset as we can**
- a merged version of following datasets: *torchgeo/ucmerced*, *NWPU-RESISC45*.

This is a union of categories from original datasets: *agricultural, airplane, airport, baseball diamond, basketball court,
beach, bridge, buildings, chaparral, church, circular farmland, cloud, commercial area, desert, forest, freeway, golf course,
ground track field, harbor, industrial area, intersection, island, lake, meadow, mountain, overpass, palace, parking lot,
railway, railway station, rectangular farmland, residential, river, roundabout, runway, sea ice, ship, snowberg, stadium,
 storage tanks, tennis court, terrace, thermal power station, wetland*.

More information available on our [HuggingFace page](https://huggingface.co/datasets/EmbeddingStudio/merged_remote_landscapes_v1).

Warning: Synonymous and ambiguous categories were combined (see "Merge method").

For being easily used for the demo we put all items of this dataset into public for reading AWS S3 Bucket:
* Region name: us-west-2
* Bucket name: embedding-studio-experiments
* Path to items: remote-lanscapes/items/

## Clickstream

We pre-generated a batch of clickstream sessions. To check the algorithm of generation, 
please visit our [experiments repo](https://github.com/EulerSearch/embedding_studio_experiments/blob/main).

Briefly, the generation method is:
1. For each category were generated up to 20 queries using GPT-3.5.
2. Using VIT-B-32 OpenAI CLIP, and Faiss.FlatIndexIP for each query were emulated search sessions.
3. And then for each search session out of each positive (related to a category of a given query) we pick random set 
as clicks with some probability of a mistake.

Params of emulation:
* A count of search results;
* A range of random picked positives;
* A probability of a mistake;


We put the result of generation into the public reading-available S3 repository:
* Region name: us-west-2
* Bucket name: embedding-studio-experiments
* Path to items: remote-lanscapes/clickstream
* A result of generation with different conditions were packed into different folders
* Generation params are available by path: remote-lanscapes/clickstream/{generation-id}/conditions.json
* Generated clickstreams are available by path: remote-lanscapes/clickstream/{generation-id}/sessions.json

"""

# Used constants
DEFAULT_FINE_TUNING_METHOD_NAME = "Default Fine Tuning Method"
BUCKET_NAME = "embedding-studio-experiments"
# We use the easiest one of generated inputs batch:
# 1. Minimal probability of a mistake: 0.01
# 2. 50 search results in each session
# 3. from 0.4 to 0.6 of random positives were picked
CLICKSTREAM_INFO_KEY = "remote-lanscapes/clickstream/f6816566-cac3-46ac-b5e4-0d5b76757c93/sessions.json"

logger = logging.getLogger(__name__)


def emulate_clickstream(
    connection_url: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    test_sessions_count: int = 600,
):
    """This is a function to run the order of http-requests to the clisktream storage service.

    The service has 2 types of methods: internal and external.

    External methods:
    - Create session
    - Add session results
    - Add session events
    - Get a session info by ID
    - Mark session as irrelevant (e.g. search results are completely nonsense)

    Internal methods:
    - Release batch of inputs. After this method, new sessions will be created with a new batch ID. We need this mechanism
      to run fine-tuning only on new sessions, not included in previous run. (**Important**: later we are going to provide a mechanism to customize it).
    - Get sessions related to a batch.

    New sessions are created with a last not release batch ID. Once you start fine-tuning - you need to 'release' a batch and create a new one.

        :param connection_url: EmbeddingStudio connection URL
        :param aws_access_key_id: AWS Access Key ID (default: None)
        :param aws_secret_access_key: AWS Secret Access Key (default: None)
        :param test_sessions_count: maximum sessions count that will be used for the demo
        :return:
    """
    print(
        f"Download emulated clickstream sessions from S3 Bucket: {BUCKET_NAME} by path {CLICKSTREAM_INFO_KEY}"
    )
    if aws_access_key_id is None or aws_secret_access_key is None:
        print("No specific AWS credentials, use Anonymous session")
        s3_client = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED)
        )
    else:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name="us-west-2",
        )
    clickstream = json.loads(
        download_s3_object_to_memory(
            s3_client, BUCKET_NAME, CLICKSTREAM_INFO_KEY
        )
        .getvalue()
        .decode("utf-8")
    )
    print(f"Downloaded {len(clickstream)} emulated clickstream sessions")

    print(f"Use {test_sessions_count} of {len(clickstream)} for emulation")
    for session_id, session_info in enumerate(
        tqdm(clickstream[:test_sessions_count])
    ):
        session_id = str(session_id)
        create_session_and_push_events(
            connection_url, session_id, session_info
        )

    print("Clickstream emulation is finished")


def emulate_fine_tuning(connection_url: str, mlflow_url: str):
    """Emulate one fine-tuning iteration
    :param connection_url: EmbeddingStudio connection URL
    :param mlflow_url: MLFlow connection URL
    :return:
    """

    print(
        """ We categorize sessions into batches. A 'batch' refers to new clickstream sessions that have not been utilized for fine-tuning.
    When initiating a new fine-tuning iteration, it's necessary to release the current batch.
    This allows new sessions to be assembled into the next batch, while the existing batch is being used.
    We do it before starting a fine-tuning task, just to show how to do it, but EmbeddingStudio does it on it's own.
    """
    )
    batch_id = release_batch(connection_url)
    if not batch_id:
        print("Unable to run fine-tuning")
        return

    task_id = start_fine_tuning(connection_url, mlflow_url, batch_id)
    if task_id is None:
        print("Unable to run fine-tuning")
        return

    experiment_id = None
    for attempt in range(100):
        experiment_id = get_mlflow_results_url(mlflow_url, batch_id)
        if experiment_id is None:
            time.sleep(60)

        else:
            break

    if experiment_id is None:
        print(
            "Something went wrong with experiments tracking system, please check logs"
        )
        return
    print(
        f"Experiment is started. If you want to check results: {mlflow_url}/#/experiments/{experiment_id}"
    )
    print(f"Start periodically getting the task status")
    while True:
        status = get_task_status(connection_url, task_id)
        if status is None:
            print(
                f"Can't get task {task_id} status, please check service logs"
            )
            break
        print(f"Task {task_id} is {status}")
        # Possible statuses: "pending", "processing", "done", "canceled", "error"
        if status in ["done", "pending"]:
            time.sleep(60)
            break


def parse_args():
    DEFAULT_AWS_ACCESS_KEY_ID = None
    DEFAULT_AWS_SECRET_ACCESS_KEY = None
    DEFAULT_EMBEDDING_STUDIO_URL = "http://localhost:5000"
    DEFAULT_MLFLOW_URL = "http://localhost:5005"
    DEFAULT_TEST_SESSIONS_COUNT = 600

    parser = argparse.ArgumentParser(
        description="EmbeddingStudio test script: clickstream and fine-tuning emulation"
    )
    parser.add_argument(
        "--aws-access-key-id",
        help=f"""AWS Access Key ID (default: {DEFAULT_AWS_ACCESS_KEY_ID})
Note: during the demo we will connect to public accessible bucket.
Warning: if set as None, we will use anonymous session:
* https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html
* https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html
""",
        default=DEFAULT_AWS_ACCESS_KEY_ID,
        type=str,
    )
    parser.add_argument(
        "--aws-secret-access-key",
        help=f"AWS Secret Access Key (default: {DEFAULT_AWS_SECRET_ACCESS_KEY})",
        default=DEFAULT_AWS_SECRET_ACCESS_KEY,
        type=str,
    )
    parser.add_argument(
        "-e",
        "--embedding-studio-url",
        help=f"EmbeddingStudio URL (default: {DEFAULT_EMBEDDING_STUDIO_URL})",
        default=DEFAULT_EMBEDDING_STUDIO_URL,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--mlflow-url",
        help=f"MLFlow URL (default: {DEFAULT_MLFLOW_URL})",
        default=DEFAULT_MLFLOW_URL,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--test-sessions-count",
        help=f"Count of test sessions to be used for the demo (default: {DEFAULT_TEST_SESSIONS_COUNT})",
        default=DEFAULT_TEST_SESSIONS_COUNT,
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    emulate_clickstream(
        args.embedding_studio_url,
        args.aws_access_key_id,
        args.aws_secret_access_key,
        args.test_sessions_count,
    )
    emulate_fine_tuning(args.embedding_studio_url, args.mlflow_url)
