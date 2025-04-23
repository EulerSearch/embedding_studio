from embedding_studio.data_storage.loaders.sql.sql_item_meta import SQLFileMeta


class PgsqlFileMeta(SQLFileMeta):
    """A metadata class for PostgreSQL file items.

    This class extends SQLFileMeta to provide metadata functionality specific to
    PostgreSQL stored files or records. It contains information about the stored object
    such as its ID and other relevant metadata.

    :param object_id: The unique identifier of the object in the PostgreSQL database
    :return: A new instance of PgsqlFileMeta
    """

    ...
