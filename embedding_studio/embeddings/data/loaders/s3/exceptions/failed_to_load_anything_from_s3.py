class FailedToLoadAnythingFromAWSS3(Exception):
    def __init__(self):
        super(FailedToLoadAnythingFromAWSS3, self).__init__(
            f"Failed to load any file from AWS S3"
        )
