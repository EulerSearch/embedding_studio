from embedding_studio.models.embeddings.models import (
    EmbeddingModelInfo,
    HnswParameters,
    MetricAggregationType,
    MetricType,
)


class EmbeddingModelFactory:
    def __init__(
        self,
        name: str,
        dimensions: int,
        metric_type: MetricType,
        metric_aggregation_type: MetricAggregationType = MetricAggregationType.MIN,
        m: int = 16,
        ef_construction: int = 64,
    ):
        self.name = name
        self.dimensions = dimensions
        self.metric_type = metric_type
        self.metric_aggregation_type = metric_aggregation_type
        self.m = m
        self.ef_construction = ef_construction

    def create_embedding_model_instance(self, id: str) -> EmbeddingModelInfo:
        hnsw_params = HnswParameters(
            m=self.m, ef_construction=self.ef_construction
        )
        return EmbeddingModelInfo(
            name=self.name,
            id=id,
            dimensions=self.dimensions,
            metric_type=self.metric_type,
            metric_aggregation_type=self.metric_aggregation_type,
            hnsw=hnsw_params,
        )
