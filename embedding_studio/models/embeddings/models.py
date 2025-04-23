from enum import Enum

from pydantic import BaseModel


class MetricType(str, Enum):
    """
    MetricType: An enum defining vector distance metrics (Euclidean, Cosine, Dot
    product) used to calculate similarity between vectors. Determines how vector
    distances are computed during similarity searches.
    """

    EUCLID = "euclid"
    COSINE = "cosine"
    DOT = "dot"


class MetricAggregationType(str, Enum):
    """
    MetricAggregationType: An enum specifying how to aggregate distance metrics when
    an object has multiple vectors (MIN or AVG). Controls how similarity scores are
    combined when comparing multi-vector objects.
    """

    MIN = "min"
    AVG = "avg"


class HnswParameters(BaseModel):
    """
    HnswParameters: Configures the Hierarchical Navigable Small World (HNSW) graph
    index parameters like m (maximum connections per node) and ef_construction
    (search width during index building). Tunes the performance and accuracy
    tradeoffs of vector searches.
    """

    m: int = 16
    ef_construction: int = 64


class SearchIndexInfo(BaseModel):
    """
    SearchIndexInfo: Contains configuration for a vector search index, including
    dimensions, metric type, aggregation method, and HNSW parameters. Provides the
    technical specifications for how vectors are indexed and searched.
    """

    dimensions: int
    metric_type: MetricType = MetricType.COSINE
    metric_aggregation_type: MetricAggregationType = MetricAggregationType.MIN
    hnsw: HnswParameters = HnswParameters()


class EmbeddingModelInfo(SearchIndexInfo):
    """
    EmbeddingModelInfo: Extends SearchIndexInfo with model name and ID, connecting
    vector parameters to specific embedding models. Links the vector database to the
    models that generate the embeddings.
    """

    name: str
    id: str
