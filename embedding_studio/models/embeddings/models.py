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

    def __str__(self):
        return f"m:{self.m}-ef:{self.ef_construction}"


class SearchIndexInfo(BaseModel):
    dimensions: int
    metric_type: MetricType = MetricType.COSINE
    metric_aggregation_type: MetricAggregationType = MetricAggregationType.MIN
    hnsw: HnswParameters = HnswParameters()

    @property
    def full_name(self) -> str:
        return f"{self.name}_{self.id}"



class EmbeddingModelInfo(BaseModel):
    name: str
    id: str

    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.id}"
