from embedding_studio.data_storage.loaders.cloud_storage.bucket_file_meta import (
    BucketFileMeta,
)


class GCPFileMeta(BucketFileMeta):
    """
    Metadata for files stored in Google Cloud Platform storage.

    This class extends BucketFileMeta to represent metadata for files
    stored in GCP Cloud Storage buckets. It inherits all properties
    and methods from BucketFileMeta without adding additional functionality.
    """

    ...
