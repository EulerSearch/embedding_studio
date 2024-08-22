class VectorDbError(Exception):
    pass


class CollectionNotFoundError(VectorDbError):
    def __init__(self, collection_id: str):
        super(CollectionNotFoundError, self).__init__(
            f"Collection with id={collection_id} does not exist"
        )


class CreateCollectionConflictError(VectorDbError):
    def __init__(self, model_passed, model_used):
        super(CreateCollectionConflictError, self).__init__(
            f"Collection already created with another model: "
            f"model_used={model_used}, model_passed={model_passed}"
        )


class LockAcquisitionError(Exception):
    """
    Exception raised when a lock cannot be acquired after multiple attempts.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"LockAcquisitionError: {self.message}"


class DeleteBlueCollectionError(VectorDbError):
    def __init__(
        self,
    ):
        super(DeleteBlueCollectionError, self).__init__(
            f"Blue collection can't be deleted"
        )
