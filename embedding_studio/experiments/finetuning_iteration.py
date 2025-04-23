from pydantic import BaseModel

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPERIMENT_PREFIX = "iteration"


class FineTuningIteration(BaseModel):
    """
    Fine-tuning iteration.

    Represents a specific iteration of the fine-tuning process, including batch ID,
    run ID, and plugin name information.

    :param batch_id: Session batch ID
    :param run_id: Starting model run ID
    :param plugin_name: Name of tuned embedding
    :return: An instance of FineTuningIteration
    """

    batch_id: str = ""
    run_id: str = ""
    plugin_name: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """
        Convert the iteration to a string representation.

        Format: "{plugin_name} / {EXPERIMENT_PREFIX} / {run_id} / {batch_id}"

        :return: String representation of the iteration
        """
        return f"{self.plugin_name} / {EXPERIMENT_PREFIX} / {str(self.run_id)} / {str(self.batch_id)}"

    @staticmethod
    def parse(experiment_name: str) -> "FineTuningIteration":
        """
        Parse an experiment name into a FineTuningIteration object.

        Handles both initial experiments and regular iteration experiments.

        :param experiment_name: Experiment name to parse
        :return: FineTuningIteration object representing the parsed experiment
        """
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
