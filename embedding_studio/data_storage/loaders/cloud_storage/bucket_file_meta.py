import hashlib
from typing import Any, Dict, Optional

from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class BucketFileMeta(ItemMeta):
    """
    Metadata class for files stored in a bucket-based storage system (like S3).

    BucketFileMeta extends ItemMeta with bucket and file path information,
    allowing items to be uniquely identified by their location in storage.
    Optionally, an index can be provided to represent sub-items within a file.

    :param bucket: Name of the storage bucket containing the file
    :param file: Path to the file within the bucket
    :param index: Optional index for sub-items within the file (e.g., chunks or lines)
    :param object_id: Optional explicit identifier for the item
    :param payload: Optional dictionary containing additional metadata
    """

    bucket: str
    file: str
    index: Optional[int] = None
    object_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def derived_id(self) -> str:
        """
        Constructs a unique identifier using the bucket and file path. Includes index if present.

        For S3FileMeta, the `derived_id` is constructed from the S3 bucket name and file path,
        ensuring that each file can be uniquely identified even across different buckets or within
        the same bucket but different paths. If an index is provided, it is appended to differentiate
        between multiple items that may come from the same file (such as data chunks or lines).
        """
        if self.index is None:
            return f"{self.bucket}/{self.file}"
        else:
            return f"{self.bucket}/{self.file}:{self.index}"

    def __hash__(self) -> int:
        """
        Computes a hash value using SHA-256 based on the item's ID.

        This implementation provides a more robust hash function than the parent class,
        using a cryptographic hash of the ID to generate an integer value.

        :return: An integer hash value generated from the SHA-256 hash of the item's ID
        """
        sha256_hash = hashlib.sha256()
        sha256_hash.update(self.id.encode("utf-8"))
        hash_result: str = sha256_hash.hexdigest()
        hash_int = int(hash_result, 16)
        return hash_int
