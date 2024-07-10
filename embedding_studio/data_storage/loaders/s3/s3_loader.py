import io
import logging
import uuid
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Type

import boto3
from botocore import UNSIGNED
from botocore.client import BaseClient, Config
from botocore.exceptions import ClientError, EndpointConnectionError
from datasets import Dataset, Features
from pydantic import BaseModel

from embedding_studio.core.config import settings
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.downloaded_item import (
    DownloadedItem,
)
from embedding_studio.data_storage.loaders.item_meta import ItemMeta
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

    @property
    def item_meta_cls(self) -> Type[ItemMeta]:
        return S3FileMeta

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

    def _get_data_from_s3(
        self, files: List[S3FileMeta], ignore_failures: bool = True
    ) -> Iterable[Tuple[Dict, S3FileMeta]]:
        """
        Main method to retrieve data from AWS S3 using a list of file metadata objects.

        :param files: A list of S3FileMeta objects containing metadata about each file to download.
        :param ignore_failures: If True, continues with next files after a failure; otherwise, raises an exception.
        :return: An iterable of tuples, each containing the data dictionary and its corresponding S3FileMeta.
        """
        uploaded = (
            {}
        )  # Cache to store downloaded data and avoid re-downloading.
        if not files:
            logger.warning("Nothing to download")
            return

        logger.info("Connecting to AWS S3...")
        task_id = str(uuid.uuid4())
        s3_client = self._get_client(task_id)
        logger.info("Start downloading data from S3...")

        for file_meta in files:
            yield from self._process_file_meta(
                s3_client, file_meta, ignore_failures, uploaded
            )

    def _process_file_meta(
        self,
        s3_client: BaseClient,
        file_meta: S3FileMeta,
        ignore_failures: bool,
        uploaded: Dict[Tuple[str, str], Any],
    ) -> Generator[Tuple[Dict, S3FileMeta], None, None]:
        """
        Processes a single file metadata to download the file and prepare data objects.

        :param s3_client: The configured boto3 S3 client.
        :param file_meta: Metadata for a specific file to handle.
        :param ignore_failures: Controls error handling behavior.
        :param uploaded: A dictionary acting as a cache to store previously downloaded data.
        :yield: Yields tuples of data dictionary and file metadata.
        """
        try:
            item = self._download_and_get_item(s3_client, file_meta, uploaded)
            if item is None:
                logger.error(
                    f"Unable to download {file_meta.file} from {file_meta.bucket}"
                )
                return
            yield from self._yield_item_objects(item, file_meta)
        except Exception as e:
            logger.exception(
                f"Unable to download an item: {file_meta.bucket}/{file_meta.file} Exception: {str(e)}"
            )
            if not ignore_failures:
                raise

    def _download_and_get_item(
        self,
        s3_client: BaseClient,
        file_meta: S3FileMeta,
        uploaded: Dict[Tuple[str, str], Any],
    ) -> Any:
        """
        Attempts to download the file from S3 if not already downloaded and cached.

        :param s3_client: Boto3 S3 client.
        :param file_meta: Metadata of the file to download.
        :param uploaded: Cache dictionary to store and retrieve downloaded files.
        :return: Downloaded or cached item data.
        """
        cache_key = (file_meta.bucket, file_meta.file)
        if cache_key not in uploaded:
            item = self._get_item(
                self._read_from_s3(s3_client, file_meta.bucket, file_meta.file)
            )
            uploaded[cache_key] = item
        return uploaded[cache_key]

    def _yield_item_objects(
        self, item: Any, file_meta: S3FileMeta
    ) -> Generator[Tuple[Dict, S3FileMeta], None, None]:
        """
        Yields data objects based on the item structure and its metadata.

        :param item: Downloaded or retrieved item data.
        :param file_meta: Metadata associated with the item.
        :yield: Generates tuples of item data dictionary and file metadata.
        """
        if isinstance(item, list) and file_meta.index is not None:
            for subitem in item:
                yield self._create_item_object(subitem, file_meta)
        else:
            yield self._create_item_object(item, file_meta)

    def _create_item_object(
        self, item: Any, file_meta: S3FileMeta
    ) -> Tuple[Dict, S3FileMeta]:
        """
        Creates a dictionary object from the item data and includes metadata.

        :param item: The data content of the item.
        :param file_meta: The metadata of the item.
        :return: A tuple containing the item dictionary and its metadata.
        """
        item_object = {"item_id": file_meta.id}
        if self.features is None or not isinstance(item, dict):
            item_object["item"] = item
        else:
            item_object.update(item)
        return item_object, file_meta

    def _generate_dataset_from_s3(
        self, files: List[S3FileMeta]
    ) -> Iterable[Tuple[Dict, S3FileMeta]]:
        for item, _ in self._get_data_from_s3(files):
            yield item

    def load(self, items_data: List[S3FileMeta]) -> Dataset:
        return Dataset.from_generator(
            lambda: self._generate_dataset_from_s3(items_data),
            features=self.features,
        )

    def load_items(self, items_data: List[S3FileMeta]) -> List[DownloadedItem]:
        result = []
        for item_object, item_meta in self._get_data_from_s3(
            items_data, ignore_failures=False
        ):
            result.append(
                DownloadedItem(
                    id=item_object["item_id"],
                    data=item_object["item"],
                    meta=item_meta,
                )
            )

        return result

    def _load_batch_with_offset(
        self, offset: int, batch_size: int, **kwargs
    ) -> List[DownloadedItem]:
        """
        Load a batch of files from S3 starting from the given offset up to the batch size.

        :param offset: The offset from where to start loading files.
        :param batch_size: The number of files to load.
        :return: A list of downloaded items, each containing the file key (ID), its content and metadata.
        """
        logger.info("Connecting to aws s3...")
        task_id: str = str(uuid.uuid4())
        s3_client = self._get_client(task_id)

        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(
            Bucket=kwargs["bucket"],
            PaginationConfig={
                "PageSize": batch_size,
                "StartingToken": str(offset),
            },
        )

        batch = []
        try:
            for page in page_iterator:
                for item in page.get("Contents", []):
                    key = item["Key"]
                    response = s3_client.get_object(
                        Bucket=kwargs["bucket"], Key=key
                    )
                    content = io.BytesIO(response["Body"].read())
                    batch.append(
                        DownloadedItem(
                            id=key,
                            data=self._get_item(content),
                            meta=S3FileMeta(bucket=kwargs["bucket"], file=key),
                        )
                    )
                    if len(batch) >= batch_size:
                        return batch
        except ClientError as e:
            logger.error(f"Error fetching batch from S3: {e}")
            raise
        return batch

    def load_all(
        self, batch_size: int, **kwargs
    ) -> Generator[DownloadedItem, None, None]:
        """
        A generator that iteratively loads batches using the `load_batch` method.
        This allows for managing large datasets by processing them in manageable chunks.
        Each batch is yielded to the caller, which can handle or process the batch as needed.

        :param batch_size: The size of each batch to load.
        :yield: Each batch as a list of downloaded items (id, data, item_info).
        """
        for bucket in kwargs["buckets"]:
            offset = 0
            while True:
                current_batch = self._load_batch_with_offset(
                    offset, batch_size, bucket=bucket
                )
                if not current_batch:
                    break  # Stop yielding if no more data is returned.
                yield current_batch
                offset += batch_size

    def total_count(self, **kwargs) -> Optional[int]:
        return None
