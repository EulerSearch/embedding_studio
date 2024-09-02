class FailedToLoadAnythingFromGCP(Exception):
    def __init__(self):
        super(FailedToLoadAnythingFromGCP, self).__init__(
            f"Failed to load any file from GCP"
        )
