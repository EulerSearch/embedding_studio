from embedding_studio.clickstream_storage.query_item import QueryItem


class TextQueryItem(QueryItem):
    """
    A query item implementation for text-based queries.

    Extends the base QueryItem class to work specifically with text queries.

    :param text: The text content of the query
    """

    text: str

    class Config:
        arbitrary_types_allowed = True
