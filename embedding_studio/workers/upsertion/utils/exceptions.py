class DownloadException(Exception):
    """Exception raised for errors during the download process."""


class SplitException(Exception):
    """Exception raised for errors during the splitting process."""


class InferenceException(Exception):
    """Exception raised for errors during the inference process."""


class UploadException(Exception):
    """Exception raised for errors during the upload process."""
