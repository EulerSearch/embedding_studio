import io
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from datasets import Dataset, Features
from pydantic import BaseModel

from embedding_studio.core.config import settings
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.s3.exceptions.failed_to_load_anything_from_s3 import (
    FailedToLoadAnythingFromAWSS3,
)
from embedding_studio.data_storage.loaders.s3.item_meta import S3FileMeta
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)
from embedding_studio.workers.fine_tuning.utils.retry import retry_method

logger = logging.getLogger(__name__)


class AwsS3Credentials(BaseModel):
    role_arn: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    external_id: Optional[str] = None
    use_system_info: bool = False


def read_from_s3(client, bucket: str, file: str) -> io.BytesIO:
    if not isinstance(bucket, str) or len(bucket) == 0:
        raise ValueError("bucket value should be not empty string")

    if not isinstance(file, str) or len(file) == 0:
        raise ValueError("file value should be not empty string")

    outfile = io.BytesIO()
    try:
        client.download_fileobj(bucket, file, outfile)
        outfile.seek(0)
        return outfile

    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.error(f"Object {file} not found in bucket {bucket}")
            return None
        else:
            # Raise the exception for any other unexpected errors
            raise e


class AwsS3DataLoader(DataLoader):
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        **kwargs,
    ):
        """Items loader from AWS S3.

        :param retry_config: retry strategy (default: None)
        :param features: expected features (default: None)
        :param kwargs: dict data for AwsS3Credentials
        """
        super(AwsS3DataLoader, self).__init__(**kwargs)
        self.retry_config = (
            retry_config
            if retry_config
            else AwsS3DataLoader._get_default_retry_config()
        )
        self.features = features
        self.credentials = AwsS3Credentials(**kwargs)
        self.attempt_exception_types = [EndpointConnectionError]

    @staticmethod
    def _get_default_retry_config() -> RetryConfig:
        default_retry_params = RetryParams(
            max_attempts=settings.DEFAULT_MAX_ATTEMPTS,
            wait_time_seconds=settings.DEFAULT_WAIT_TIME_SECONDS,
        )

        config = RetryConfig(default_params=default_retry_params)
        config["credentials"] = RetryParams(
            max_attempts=settings.S3_READ_CREDENTIALS_ATTEMPTS,
            wait_time_seconds=settings.S3_READ_WAIT_TIME_SECONDS,
        )
        config["download_data"] = RetryParams(
            max_attempts=settings.S3_DOWNLOAD_DATA_ATTEMPTS,
            wait_time_seconds=settings.S3_DOWNLOAD_DATA_WAIT_TIME_SECONDS,
        )
        return config

    @retry_method(name="download_data")
    def _read_from_s3(self, client, bucket: str, file: str) -> Any:
        return read_from_s3(client, bucket, file)

    @retry_method(name="credentials")
    def _get_client(self, task_id: str):
        if (
            self.credentials.aws_access_key_id is None
            or self.credentials.aws_secret_access_key is None
        ) and not self.credentials.use_system_info:
            logger.warning(
                "No specific AWS credentials, use Anonymous session"
            )
            s3_client = boto3.client(
                "s3", config=Config(signature_version=UNSIGNED)
            )
        else:
            sts_client = boto3.client(
                "sts",
                aws_access_key_id=self.credentials.aws_access_key_id,
                aws_secret_access_key=self.credentials.aws_secret_access_key,
            )
            if self.credentials.external_id:
                assumed_role_object = sts_client.assume_role(
                    RoleArn=self.credentials.role_arn,
                    RoleSessionName=task_id,
                    ExternalId=self.credentials.external_id,
                )
            else:
                assumed_role_object = sts_client.assume_role(
                    RoleArn=self.credentials.role_arn,
                    RoleSessionName=task_id,
                )
            credentials = assumed_role_object["Credentials"]
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            )
        return s3_client

    def _get_item(self, file: io.BytesIO) -> Any:
        return file

    def _generate_dataset_from_s3(
        self, files: List[S3FileMeta]
    ) -> Iterable[Dict]:
        if len(files) == 0:
            logger.warning("Nothing to download")
        else:
            logger.info("Connecting to aws s3...")
            task_id: str = str(uuid.uuid4())
            try:
                s3_client = self._get_client(task_id)
                logger.info("Start downloading data from S3...")
                bad_items_count = 0
                for val in files:
                    item = None
                    try:
                        item: Any = self._get_item(
                            self._read_from_s3(s3_client, val.bucket, val.file)
                        )
                    except Exception as e:
                        logger.exception(
                            f"Unable to download an item: {val.bucket}/{val.file} Exception: {str(e)}"
                        )

                    if item is None:
                        logger.error(
                            f"Unable to download {val.file} from {val.bucket}"
                        )
                        bad_items_count += 1
                        continue

                    if isinstance(item, list):
                        for i, subitem in enumerate(item):
                            item_object = {"item_id": f"{val.id}:{i}"}
                            if self.features is None or not isinstance(
                                subitem, dict
                            ):
                                item_object["item"] = subitem
                            else:
                                item_object.update(subitem)
                            yield item_object
                    else:
                        item_object = {"item_id": val.id}
                        if self.features is None or not isinstance(item, dict):
                            item_object["item"] = item
                        else:
                            item_object.update(item)
                        yield item

                if bad_items_count == len(files):
                    raise FailedToLoadAnythingFromAWSS3()

            except Exception as err:
                logger.error(f"Failed to load dataset from s3: {err}")
                raise err

    def load(self, items_data: List[S3FileMeta]) -> Dataset:
        return Dataset.from_generator(
            lambda: self._generate_dataset_from_s3(items_data),
            features=self.features,
        )
