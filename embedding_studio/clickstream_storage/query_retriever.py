from typing import List

from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)


class QueryRetriever(object):
    """As we can't exactly predict a schema of storing queries:
    1. As dict exceptly in clickstream service
    2. As ID of a record with a dict
    3. As a path to an image

    We provide an ability to use any query item. So, a user can specify any.

    """

    def get_queries(self, inputs: List[FineTuningInput]):
        pass
