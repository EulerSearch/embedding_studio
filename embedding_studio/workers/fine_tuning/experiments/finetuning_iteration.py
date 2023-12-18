from pydantic import BaseModel

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPERIMENT_PREFIX = "iteration"


class FineTuningIteration(BaseModel):
    """Fine tuning iteration.

    :param batch_id: session batch id
    :type batch_id:  str
    :param plugin_name: name of tuned embedding (default: "")
    :type plugin_name: str
    """

    batch_id: str = ""
    plugin_name: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return (
            f"{EXPERIMENT_PREFIX} / {self.plugin_name} / " + f"{self.batch_id}"
        )
