from pydantic import BaseModel

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPERIMENT_PREFIX = "iteration"


class FineTuningIteration(BaseModel):
    """Fine-tuning iteration.

    :param batch_id: session batch id
    :param run_id: starting model run id.
    :param plugin_name: name of tuned embedding (default: "")
    """

    batch_id: str = ""
    run_id: str = ""
    plugin_name: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return f"{self.plugin_name} / {EXPERIMENT_PREFIX} / {str(self.run_id)} / {str(self.batch_id)}"
