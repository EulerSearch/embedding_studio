import hashlib

from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class S3FileMeta(ItemMeta):
    bucket: str
    file: str

    class Config:
        arbitrary_types_allowed = True

    @property
    def id(self) -> str:
        return f"{self.bucket}/{self.file}"

    def __hash__(self) -> int:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(self.id.encode("utf-8"))
        hash_result: str = sha256_hash.hexdigest()
        hash_int = int(hash_result, 16)
        return hash_int
