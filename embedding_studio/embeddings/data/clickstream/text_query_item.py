from embedding_studio.embeddings.data.clickstream.query_item import QueryItem


class TextQueryItem(QueryItem):
    text: str

    class Config:
        arbitrary_types_allowed = True
