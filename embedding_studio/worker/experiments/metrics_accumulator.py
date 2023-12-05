from typing import List, Tuple

import numpy as np


class MetricValue:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def add_prefix(self, prefix: str):
        self.name = f"{prefix}_{self.name}"
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
        :type name: str
        :param calc_mean: should accumulator calculate mean value (default: False)
        :type calc_mean: bool
        :param calc_sliding: should accumulator calculate sliding mean value (default: False)
        :type calc_sliding: bool
        :param calc_min: should accumulator calculate min value (default: False)
        :type calc_min: bool
        :param calc_max: should accumulator calculate max value (default: False)
        :type calc_max: bool
        :param window_size: size of sliding window (default: 10)
        :type window_size: int
        """
        self.name = name
        self.calc_mean = calc_mean
        self.calc_sliding = calc_sliding
        self.calc_min = calc_min
        self.calc_max = calc_max
        self.window_size = window_size
        self._values = []

    def clear(self):
        """Clear accumulator"""
        self._values = []

    def accumulate(self, value: MetricValue) -> List[Tuple[str, float]]:
        """Add metric value to an accumulator.

        :param value: metric to be accumulated
        :type value:  MetricValue
        :return: aggregations
        :rtype: List[Tuple[str, float]]
        """
        if self.name == value.name:
            self._values.append(value.value)

            return self.aggregate()

        return []

    def aggregate(self) -> List[Tuple[str, float]]:
        """Aggregate accumulated metrics

        :return: metric aggregations (last, mean, sliding, min, max)
        :rtype:  List[Tuple[str, float]]
        """
        aggregations: List[Tuple[str, float]] = []
        if len(self._values) > 0:
            aggregations.append((self.name, self._values[-1]))
            if self.calc_mean:
                aggregations.append(
                    (f"mean_{self.name}", np.mean(self._values))
                )

            if self.calc_sliding:
                slide_value = float(
                    np.mean(self._values)
                    if len(self._values) < self.window_size
                    else np.mean(self._values[-self.window_size :])
                )
                aggregations.append((f"sliding_{self.name}", slide_value))

            if self.calc_min:
                aggregations.append((f"min_{self.name}", np.min(self._values)))

            if self.calc_max:
                aggregations.append((f"max_{self.name}", np.max(self._values)))

        return aggregations
