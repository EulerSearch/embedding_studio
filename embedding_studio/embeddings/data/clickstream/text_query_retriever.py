from embedding_studio.embeddings.data.clickstream.query_retriever import (
    QueryRetriever,
)
from embedding_studio.embeddings.data.clickstream.text_query_item import (
    TextQueryItem,
)


class TextQueryRetriever(QueryRetriever):
    def __call__(self, query: TextQueryItem) -> str:
        return query.text
