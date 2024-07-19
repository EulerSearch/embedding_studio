import hashlib
from typing import Any, Dict, Optional

from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class BucketFileMeta(ItemMeta):
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
        sha256_hash = hashlib.sha256()
        sha256_hash.update(self.id.encode("utf-8"))
        hash_result: str = sha256_hash.hexdigest()
        hash_int = int(hash_result, 16)
        return hash_int
