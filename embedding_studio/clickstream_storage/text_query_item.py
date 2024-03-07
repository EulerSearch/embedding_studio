from embedding_studio.clickstream_storage.query_item import QueryItem


class TextQueryItem(QueryItem):
    text: str

    class Config:
        arbitrary_types_allowed = True
