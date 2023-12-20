import hashlib
from enum import Enum
from typing import List

from pydantic import BaseModel, root_validator, validator


class ExamplesType(Enum):
    medium_all = 1
    hard_all = 2
    soft_all = 3
    medium_positive = 4
    hard_positive = 5
    soft_positive = 6
    medium_negative = 7
    hard_negative = 8
    soft_negative = 9
    negative_only = 10
    all_examples = 11


class FineTuningParams(BaseModel):
    """Params of fine-tuning procedure

    :param num_fixed_layers: number of fixed embeddings layers
    :param query_lr: learning rate of query model optimizer
    :param items_lr: learning rate of items model optimizer
    :param query_weight_decay: weight decay of query model optimizer
    :param items_weight_decay: weight decay of items model optimizer
    :param margin: margin from MarginRankingLoss
    :param not_irrelevant_only: use only not irrelevant sessions
    :param negative_downsampling: ratio of negative samples to be used
    :param min_abs_difference_threshold: filter out soft pairs abs(neg_dist - pos_dist) < small value (default: 0.0)
    :param max_abs_difference_threshold: filter out hard pairs abs(neg_dist - pos_dist) > huge value (default: 1.0)
    :param examples_order: order of passing examples to a trainer (default: None)
    """

    num_fixed_layers: int
    query_lr: float
    items_lr: float
    query_weight_decay: float
    items_weight_decay: float
    margin: float
    not_irrelevant_only: bool
    negative_downsampling: float
    min_abs_difference_threshold: float = 0.0
    max_abs_difference_threshold: float = 1.0
    examples_order: List[ExamplesType] = [ExamplesType.all_examples]

    class Config:
        arbitrary_types_allowed = True

    @validator("examples_order", pre=True, always=True)
    def validate_examples_order(cls, value):
        if isinstance(value, str):
            value = list(map(int, value.split(",")))
        elif isinstance(value, tuple):
            value = list(value)
        return [ExamplesType(v) for v in value]

    @validator("items_lr", "query_lr", pre=True, always=True)
    def validate_positive_float(cls, value):
        if not (isinstance(value, float) and value > 0):
            raise ValueError(f"{value} must be a positive float")
        return value

    @validator(
        "items_weight_decay", "query_weight_decay", pre=True, always=True
    )
    def validate_non_negative_float(cls, value):
        if not (isinstance(value, float) and value >= 0):
            raise ValueError(f"{value} must be a non-negative float")
        return value

    @validator("margin", pre=True, always=True)
    def validate_non_negative_float_margin(cls, value):
        if not (isinstance(value, float) and value >= 0):
            raise ValueError(f"{value} must be a non-negative float")
        return value

    @validator("num_fixed_layers", pre=True, always=True)
    def validate_non_negative_int(cls, value):
        if not (isinstance(value, int) and value >= 0):
            raise ValueError(f"{value} must be a non-negative integer")
        return value

    @root_validator(skip_on_failure=True)
    def validate_example_order(cls, values):
        examples_order = values.get("examples_order")
        if examples_order:
            if isinstance(examples_order, str):
                examples_order = list(map(int, examples_order.split(",")))
            elif isinstance(examples_order, tuple):
                examples_order = list(examples_order)
            values["examples_order"] = [
                ExamplesType(v) for v in examples_order
            ]
        return values

    @property
    def id(self) -> str:
        # Convert the value to bytes (assuming it's a string)
        value_bytes: bytes = str(self).encode("utf-8")

        # Create a hash object
        hash_object = hashlib.sha256()

        # Update the hash object with the value
        hash_object.update(value_bytes)

        # Get the hexadecimal representation of the hash
        unique_id: str = hash_object.hexdigest()

        return unique_id

    def __str__(self) -> str:
        vals: List[str] = []
        for key, value in dict(self).items():
            value = (
                ",".join(map(str, value)) if isinstance(value, list) else value
            )
            vals.append(f"{key}: {value}")

        return " / ".join(vals)
