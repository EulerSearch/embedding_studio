import fcntl


def acquire_lock(lock_file_path: str):
    """
    Acquire an exclusive lock on the given file.
    """
    lock_file = open(lock_file_path, "w")
    fcntl.lockf(lock_file, fcntl.LOCK_EX)
    return lock_file


def release_lock(lock_file: str):
    """
    Release the lock on the given file.
    """
    fcntl.lockf(lock_file, fcntl.LOCK_UN)
    lock_file.close()
