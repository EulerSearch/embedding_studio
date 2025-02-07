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

    @staticmethod
    def parse(experiment_name: str) -> "FineTuningIteration":
        split_parts = experiment_name.split(" / ")
        if "initial" in split_parts:
            if len(split_parts) != 3:
                raise ValueError(
                    "Experiment name does not follow the pattern."
                )

            return FineTuningIteration(
                plugin_name=split_parts[0],
            )

        else:
            if len(split_parts) != 4 or split_parts[1] != EXPERIMENT_PREFIX:
                raise ValueError(
                    "Experiment name does not follow the pattern."
                )

            return FineTuningIteration(
                plugin_name=split_parts[0],
                run_id=split_parts[2],
                batch_id=split_parts[3],
            )
