class BestParamsNotFoundError(Exception):
    def __init__(self, experiment_id: str):
        super(BestParamsNotFoundError, self).__init__(
            f"Cannot retrieve best params list "
            f"for the experiment with ID {experiment_id}"
        )
        self.experiment_id = experiment_id


class ModelNotFoundError(Exception):
    def __init__(self, run_id: str):
        super(ModelNotFoundError, self).__init__(
            f"Cannot retrieve a model for the run with ID {run_id}"
        )
        self.run_id = run_id


class ParamsNotFoundError(Exception):
    def __init__(self, run_id: str):
        super(ParamsNotFoundError, self).__init__(
            f"Cannot retrieve fine-tuning params for the run with ID {run_id}"
        )
        self.run_id = run_id
