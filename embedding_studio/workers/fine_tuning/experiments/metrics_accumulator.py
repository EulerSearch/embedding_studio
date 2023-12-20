from typing import List, Tuple

import numpy as np


class MetricValue:
    def __init__(self, name: str, value: float):
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("MetricValue's name should not be empty")
        self._name = name

        if not isinstance(value, float):
            raise ValueError("MetricValue's value should not be numeric")
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> float:
        return self._value

    def add_prefix(self, prefix: str):
        self._name = f"{prefix}_{self._name}"
        return self


class MetricsAccumulator:
    def __init__(
        self,
        name: str,
        calc_mean: bool = False,
        calc_sliding: bool = False,
        calc_min: bool = False,
        calc_max: bool = False,
        window_size: int = 10,
    ):
        """Accumulator of metric values + calculator of aggregations like mean, max, min, sliding_mean.

        :param name: metric name (metrics with other name will be ignored)
        :param calc_mean: should accumulator calculate mean value (default: False)
        :param calc_sliding: should accumulator calculate sliding mean value (default: False)
        :param calc_min: should accumulator calculate min value (default: False)
        :param calc_max: should accumulator calculate max value (default: False)
        :param window_size: size of sliding window (default: 10)
        """
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("MetricsAccumulator's name should not be empty")

        self._name = name

        if not isinstance(calc_mean, bool):
            raise ValueError("calc_mean value should be bool")
        self._calc_mean = calc_mean

        if not isinstance(calc_sliding, bool):
            raise ValueError("calc_sliding value should be bool")
        self._calc_sliding = calc_sliding

        if not isinstance(calc_min, bool):
            raise ValueError("calc_min value should be bool")
        self._calc_min = calc_min

        if not isinstance(calc_max, bool):
            raise ValueError("calc_max value should be bool")
        self._calc_max = calc_max

        if not isinstance(window_size, int) or window_size <= 1:
            raise ValueError(
                "window_size value should be integer with value more than 1"
            )

        self._window_size = window_size
        self._values = []

    @property
    def name(self) -> str:
        return self._name

    def clear(self):
        """Clear accumulator"""
        self._values = []

    def accumulate(self, value: MetricValue) -> List[Tuple[str, float]]:
        """Add metric value to an accumulator.

        :param value: metric to be accumulated
        :return: aggregations
        """
        if self.name == value.name:
            self._values.append(value.value)

            return self.aggregate()

        return []

    def aggregate(self) -> List[Tuple[str, float]]:
        """Aggregate accumulated metrics

        :return: metric aggregations (last, mean, sliding, min, max)
        """
        aggregations: List[Tuple[str, float]] = []
        if len(self._values) > 0:
            aggregations.append((self.name, self._values[-1]))
            if self._calc_mean:
                aggregations.append(
                    (f"mean_{self.name}", float(np.mean(self._values)))
                )

            if self._calc_sliding:
                slide_value = float(
                    np.mean(self._values)
                    if len(self._values) < self._window_size
                    else np.mean(self._values[-self._window_size :])
                )
                aggregations.append((f"sliding_{self.name}", slide_value))

            if self._calc_min:
                aggregations.append((f"min_{self.name}", np.min(self._values)))

            if self._calc_max:
                aggregations.append((f"max_{self.name}", np.max(self._values)))

        return aggregations
