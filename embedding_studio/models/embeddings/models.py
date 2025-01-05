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


class SearchIndexInfo(BaseModel):
    dimensions: int
    metric_type: MetricType = MetricType.COSINE
    metric_aggregation_type: MetricAggregationType = MetricAggregationType.MIN
    hnsw: HnswParameters = HnswParameters()


class EmbeddingModelInfo(BaseModel):
    name: str
    id: str

    @property
    def full_name(self) -> str:
        return f"{self.name}_{self.id}"
