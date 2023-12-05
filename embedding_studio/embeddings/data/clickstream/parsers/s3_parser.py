from embedding_studio.embeddings.data.clickstream.parsers.parser import (
    ClickstreamParser,
)
from embedding_studio.embeddings.data.loaders.s3.item_meta import S3FileMeta


class AWSS3ClickstreamParser(ClickstreamParser):
    def __init__(
        self, query_item_type: type, search_result_type: type, event_type: type
    ):
        super(AWSS3ClickstreamParser, self).__init__(
            query_item_type, search_result_type, S3FileMeta, event_type
        )
