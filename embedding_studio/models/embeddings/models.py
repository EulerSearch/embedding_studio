from enum import Enum

from pydantic import BaseModel


class MetricType(str, Enum):
    EUCLID = "euclid"
    COSINE = "cosine"
    DOT = "dot"


class MetricAggregationType(str, Enum):
    MIN = "min"
    AVG = "avg"


class HnswParameters(BaseModel):
    m: int = 16
    ef_construction: int = 64


class EmbeddingModel(BaseModel):
    name: str
    id: str
    dimensions: int
    metric_type: MetricType
    metric_aggregation_type: MetricAggregationType = MetricAggregationType.MIN
    hnsw: HnswParameters = HnswParameters()
