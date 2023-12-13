class MaxAttemptsReachedException(Exception):
    def __init__(self, attempts: int):
        super(MaxAttemptsReachedException, self).__init__(
            f"Reached maximum number of attempts: {attempts}"
        )
