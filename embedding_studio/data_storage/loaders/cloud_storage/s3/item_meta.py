from embedding_studio.data_storage.loaders.cloud_storage.bucket_file_meta import (
    BucketFileMeta,
)


class S3FileMeta(BucketFileMeta):
    """
    Metadata for an AWS S3 file.

    Extends the BucketFileMeta class with S3-specific functionality.
    This class contains information about an S3 file, including its bucket, path, and other metadata.
    """

    ...
