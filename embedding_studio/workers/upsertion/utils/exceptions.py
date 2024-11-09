class DownloadException(Exception):
    """Exception raised for errors during the download process."""


class SplitException(Exception):
    """Exception raised for errors during the splitting process."""


class InferenceException(Exception):
    """Exception raised for errors during the inference process."""


class UploadException(Exception):
    """Exception raised for errors during the upload process."""


class UpsertionException(Exception):
    """Exception raised for errors during the other stage of upsertion process."""


class DeletionException(Exception):
    """Exception raised for errors during the other stage of deletion process."""


class ReindexException(Exception):
    """Exception raised for errors during the other stage of reindex process."""
