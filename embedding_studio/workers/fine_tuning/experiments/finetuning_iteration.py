from datetime import datetime

from pydantic import BaseModel

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPERIMENT_PREFIX = "iteration"


class FineTuningIteration(BaseModel):
    """Fine tuning iteration.

    :param start: start datetime of used search sessions period
    :type start:  Union[int, datetime]
    :param end: end datetime of used search sessions period
    :type end:  Union[int, datetime]
    :param plugin_name: name of tuned embedding (default: "")
    :type plugin_name: str
    """

    start: datetime
    end: datetime
    plugin_name: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return (
            f"{EXPERIMENT_PREFIX} / {self.plugin_name} / "
            + f"{self.start.strftime(DATE_FORMAT)}-{self.end.strftime(DATE_FORMAT)}"
        )
