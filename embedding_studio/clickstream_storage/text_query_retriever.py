from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.clickstream_storage.text_query_item import TextQueryItem


class TextQueryRetriever(QueryRetriever):
    def __call__(self, query: TextQueryItem) -> str:
        if not hasattr(query, "dict"):
            raise ValueError(f"Query object does not have dict attribute")
        return query.text
