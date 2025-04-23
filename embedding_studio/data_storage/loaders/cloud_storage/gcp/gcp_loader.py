import io
import logging
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Type

from datasets import Dataset, Features
from google.cloud import storage
from google.cloud.storage import Blob
from pydantic import BaseModel

from embedding_studio.core.config import settings
from embedding_studio.data_storage.loaders.cloud_storage.gcp.item_meta import (
    GCPFileMeta,
)
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.downloaded_item import (
    DownloadedItem,
)
from embedding_studio.data_storage.loaders.item_meta import ItemMeta
from embedding_studio.utils.retry import retry_method
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)

logger = logging.getLogger(__name__)


class GCPCredentials(BaseModel):
    """
    Configuration model for GCP authentication credentials.

    This class defines the required parameters for authenticating with
    Google Cloud Platform services.

    Attributes:
        project_id: Optional identifier for the GCP project.
        credentials_path: Optional path to the service account credentials file.
        use_system_info: Whether to use system-provided credentials instead of explicit ones.
    """

    project_id: Optional[str] = None
    credentials_path: Optional[str] = None
    use_system_info: bool = False


def read_from_gcp(
    bucket: str, file: str, client: storage.Client
) -> io.BytesIO:
    """
    Read a file from GCP Cloud Storage.

    :param bucket: Name of the GCP bucket.
    :param file: File name to be downloaded.
    :param client: Initialized GCP storage client.
    :return: io.BytesIO object containing the file's contents.
    """
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("bucket value should be not empty string")

    if not isinstance(file, str) or not file:
        raise ValueError("file value should be not empty string")

    blob = client.bucket(bucket).blob(file)
    outfile = io.BytesIO()
    blob.download_to_file(outfile)
    outfile.seek(0)
    return outfile


class GCPDataLoader(DataLoader):
    """
    Items loader from GCP Cloud Storage.
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        features: Optional[Features] = None,
        **kwargs,
    ):
        """
        Initialize the DataLoader with retry configurations and expected features.

        :param retry_config: Retry strategy configuration.
        :param features: Schema for data to be loaded.
        :param kwargs: Keyword arguments for GcpCredentials.
        """
        super(GCPDataLoader, self).__init__(**kwargs)
        self.retry_config = (
            retry_config
            if retry_config
            else GCPDataLoader._get_default_retry_config()
        )
        self.features = features
        self.credentials = GCPCredentials(**kwargs)
        self.attempt_exception_types = [
            Exception
        ]  # Update based on GCP exceptions

    @property
    def item_meta_cls(self) -> Type[ItemMeta]:
        """
        Return the class used for item metadata.

        :return: The GCPFileMeta class used for representing file metadata.
        """
        return GCPFileMeta

    @staticmethod
    def _get_default_retry_config() -> RetryConfig:
        """
        Define the default retry configuration for operations.

        :return: Default retry configuration.
        """
        default_retry_params = RetryParams(
            max_attempts=settings.DEFAULT_MAX_ATTEMPTS,
            wait_time_seconds=settings.DEFAULT_WAIT_TIME_SECONDS,
        )

        config = RetryConfig(default_params=default_retry_params)
        config["credentials"] = RetryParams(
            max_attempts=settings.GCP_READ_CREDENTIALS_ATTEMPTS,
            wait_time_seconds=settings.GCP_READ_WAIT_TIME_SECONDS,
        )
        config["download_data"] = RetryParams(
            max_attempts=settings.GCP_DOWNLOAD_DATA_ATTEMPTS,
            wait_time_seconds=settings.GCP_DOWNLOAD_DATA_WAIT_TIME_SECONDS,
        )
        return config

    @retry_method(name="download_data")
    def _read_from_gcp(self, client, bucket: str, file: str) -> Any:
        """
        Wrapper for retrying reading a file from GCP Cloud Storage.

        :param client: Initialized GCP storage client.
        :param bucket: Name of the bucket.
        :param file: Name of the file to download.
        :return: File content as io.BytesIO.
        """
        return read_from_gcp(bucket, file, client)

    @retry_method(name="credentials")
    def _get_client(self) -> storage.Client:
        """
        Get a GCP storage client, either with explicit credentials or default system credentials.

        :return: Initialized GCP storage client.
        """
        if self.credentials.use_system_info:
            return storage.Client.create_anonymous_client()
        else:
            return storage.Client.from_service_account_json(
                self.credentials.credentials_path
            )

    def _get_item(self, file: io.BytesIO) -> Any:
        """
        Retrieve item data from the file object.

        :param file: File object to extract data from.
        :return: Data extracted from the file.
        """
        return file  # Convert or parse file content as needed

    def _get_data_from_gcp(
        self, files: List[GCPFileMeta], ignore_failures: bool = True
    ) -> Iterable[Tuple[Dict, GCPFileMeta]]:
        """
        Main method to retrieve data from GCP using a list of file metadata objects.

        :param files: A list of GCPFileMeta objects containing metadata about each file to download.
        :param ignore_failures: If True, continues with next files after a failure; otherwise, raises an exception.
        :return: An iterable of tuples, each containing the data dictionary and its corresponding GCPFileMeta.
        """
        uploaded = (
            {}
        )  # Cache to store downloaded data and avoid re-downloading.
        if not files:
            logger.warning("Nothing to download")
            return

        logger.info("Connecting to GCP Cloud Storage...")
        try:
            gcp_client = self._get_client()
            logger.info("Start downloading data from GCP...")

            for file_meta in files:
                yield from self._process_file_meta(
                    gcp_client, file_meta, ignore_failures, uploaded
                )

        except Exception as err:
            logger.error(f"Failed to load dataset from GCP: {err}")
            raise err

    def _process_file_meta(
        self,
        gcp_client: storage.Client,
        file_meta: GCPFileMeta,
        ignore_failures: bool,
        uploaded: Dict[Tuple[str, str], Any],
    ) -> Generator[Tuple[Dict, GCPFileMeta], None, None]:
        """
        Processes a single file metadata to download the file and prepare data objects.

        :param gcp_client: The configured GCP storage client.
        :param file_meta: Metadata for a specific file to handle.
        :param ignore_failures: Controls error handling behavior.
        :param uploaded: A dictionary acting as a cache to store previously downloaded data.
        :yield: Yields tuples of data dictionary and file metadata.
        """
        try:
            item = self._download_and_get_item(gcp_client, file_meta, uploaded)
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
        gcp_client: storage.Client,
        file_meta: GCPFileMeta,
        uploaded: Dict[Tuple[str, str], Any],
    ) -> Any:
        """
        Attempts to download the file from GCP if not already downloaded and cached.

        :param gcp_client: GCP storage client.
        :param file_meta: Metadata of the file to download.
        :param uploaded: Cache dictionary to store and retrieve downloaded files.
        :return: Downloaded or cached item data.
        """
        cache_key = (file_meta.bucket, file_meta.file)
        if cache_key not in uploaded:
            item = self._get_item(
                self._read_from_gcp(
                    gcp_client, file_meta.bucket, file_meta.file
                )
            )
            uploaded[cache_key] = item
        return uploaded[cache_key]

    def _yield_item_objects(
        self, item: Any, file_meta: GCPFileMeta
    ) -> Generator[Tuple[Dict, GCPFileMeta], None, None]:
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
        self, item: Any, file_meta: GCPFileMeta
    ) -> Tuple[Dict, GCPFileMeta]:
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

    def load(self, items_data: List[GCPFileMeta]) -> Dataset:
        """
        Load data as a Hugging Face Dataset from GCP Cloud Storage.

        :param items_data: List of GCPFileMeta for data to be loaded.
        :return: Dataset containing the loaded data.
        """
        return Dataset.from_generator(
            lambda: self._generate_dataset_from_gcp(items_data),
            features=self.features,
        )

    def _generate_dataset_from_gcp(
        self, files: List[GCPFileMeta]
    ) -> Iterable[Tuple[Dict, GCPFileMeta]]:
        """
        Generate dataset entries from GCP data.

        :param files: List of file metadata from GCP.
        :return: Iterable of data entries.
        """
        for item, _ in self._get_data_from_gcp(files):
            yield item

    def load_items(
        self, items_data: List[GCPFileMeta]
    ) -> List[DownloadedItem]:
        """
        Load specific items from GCP Cloud Storage and return them as a list of DownloadedItem objects.

        :param items_data: List of GCPFileMeta data specifying which items to load.
        :return: List of DownloadedItem objects with loaded data and metadata.
        """
        result = []
        for item_object, item_meta in self._get_data_from_gcp(
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

    @retry_method(name="download_data")
    def _download_blob(self, blob: Blob) -> Any:
        content = io.BytesIO()
        blob.download_to_file(content)
        content.seek(0)

        return content

    @retry_method(name="download_data")
    def _list_blobs(self, bucket: str) -> List[Blob]:
        """
        List all blobs in a specified GCP Cloud Storage bucket.

        :param bucket: The name of the GCP Cloud Storage bucket.
        :return: A list of Blob objects in the specified bucket.
        """
        logger.info("Connecting to GCP Cloud Storage...")
        gcp_client = self._get_client()

        bucket = gcp_client.bucket(bucket)
        return list(bucket.list_blobs())

    def _load_batch_with_offset(
        self, offset: int, batch_size: int, **kwargs
    ) -> List[DownloadedItem]:
        """
        Load a batch of files from GCP starting from a given offset up to the specified batch size.

        :param offset: The offset from where to start loading files.
        :param batch_size: The number of files to load.
        :param kwargs: Additional keyword arguments including the bucket name.
        :return: A list of DownloadedItem, each containing the file key (ID), its content and metadata.
        """

        blobs = self._list_blobs(kwargs["bucket"])
        batch = []

        for index, blob in enumerate(blobs[offset : offset + batch_size]):
            try:
                content = self._download_blob(blob)

                batch.append(
                    DownloadedItem(
                        id=blob.name,
                        data=self._get_item(content),
                        meta=GCPFileMeta(
                            bucket=kwargs["bucket"], file=blob.name
                        ),
                    )
                )
            except Exception:
                # TODO: pass failed_ids and related exceptions to the worker status
                logger.exception(
                    f"Error fetching batch item {blob.name} from GCP"
                )

        return batch

    def load_all(
        self, batch_size: int, **kwargs
    ) -> Generator[DownloadedItem, None, None]:
        """
        A generator that iteratively loads batches using the `_load_batch_with_offset` method.
        This allows for managing large datasets by processing them in manageable chunks.

        :param batch_size: The size of each batch to load.
        :param kwargs: Additional parameters including bucket names.
        :yield: Each batch as a list of DownloadedItem.
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
                offset += len(current_batch)

    def total_count(self, **kwargs) -> Optional[int]:
        """
        Optionally calculate the total number of files in the specified buckets.

        :param kwargs: Additional parameters including bucket names.
        :return: Total count of files across all specified buckets, if applicable.
        """
        total = 0
        gcp_client = self._get_client()
        for bucket_name in kwargs["buckets"]:
            gcp_client.bucket(bucket_name)
            total += sum(1 for _ in self._list_blobs(kwargs["bucket"]))
        return total
