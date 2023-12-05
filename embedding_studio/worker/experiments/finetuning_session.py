from datetime import datetime

from pydantic import BaseModel

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPERIMENT_PREFIX = "session"


class FineTuningSession(BaseModel):
    """Fine tuning session.

    :param timestamp: time of running fine-tuning procedure
    :type timestamp:  Union[int, datetime]
    :param start: start datetime of used search sessions period
    :type start:  Union[int, datetime]
    :param end: end datetime of used search sessions period
    :type end:  Union[int, datetime]
    :param model_name: name of tuned embedding (default: "")
    :type model_name: str
    """

    timestamp: datetime
    start: datetime
    end: datetime
    model_name: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return (
            f"{EXPERIMENT_PREFIX} / model_name: {self.model_name} / timestamp: {self.timestamp.strftime(DATE_FORMAT)} / "
            + f"clicks period: {self.start.strftime(DATE_FORMAT)} - {self.end.strftime(DATE_FORMAT)}"
        )
