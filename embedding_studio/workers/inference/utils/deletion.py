def handle_deletion(task_id: str):
    """
    Handles deletion of a deployed embedding model from Triton Inference Server.

    This includes:
    - Validating the deletion task and model existence
    - Ensuring the model uses a supported plugin and is not currently used in reindexing
    - Acquiring a lock to prevent concurrent deletions
    - Deleting both "query" and (optionally) "items" models from the Triton model repository
    - Updating task status accordingly

    :param task_id: ID of the model deletion task to process
    """
    model_repo = os.getenv("MODEL_REPOSITORY", os.getcwd())

    # Fetch deletion task
    task = context.model_deletion_task.get(id=task_id)
    if not task:
        raise InferenceWorkerException(
            f"Deployment task with ID `{task_id}` not found"
        )

    # Validate model iteration
    iteration = context.mlflow_client.get_iteration_by_id(
        task.embedding_model_id
    )
    if iteration is None:
        task.status = TaskStatus.failed
        context.model_deletion_task.update(obj=task)
        message = f"Can not find iteration for embedding model {task.embedding_model_id}"
        logger.error(message)
        raise InferenceWorkerException(message)

    # Check if plugin is supported
    if iteration.plugin_name not in settings.INFERENCE_USED_PLUGINS:
        task.status = TaskStatus.refused
        context.model_deletion_task.update(obj=task)
        raise InferenceWorkerException(
            f"Passed plugin is not in the used plugin list"
            f' ({", ".join(settings.INFERENCE_USED_PLUGINS)}).'
        )

    model_id = task.embedding_model_id

    # Check if the model is in use by a reindexing task
    if context.reindex_locks.get_by_dst_model_id(model_id) is not None:
        task.status = TaskStatus.refused
        context.model_deletion_task.update(obj=task)
        raise InferenceWorkerException(
            f"Can not delete the model with ID [{task.embedding_model_id}]: it's being used in reindexing."
        )

    # Locking to prevent concurrent deletions
    temp_dir = tempfile.gettempdir()
    lock_file_path = os.path.join(
        temp_dir, f"deployment_lock_{task.embedding_model_id}.lock"
    )
    lock_file = acquire_lock(lock_file_path)
    try:
        task.status = TaskStatus.processing
        context.model_deletion_task.update(obj=task)

        # Re-check iteration in case of race conditions
        iteration = context.mlflow_client.get_iteration_by_id(
            task.embedding_model_id
        )
        if iteration is None:
            task.status = TaskStatus.failed
            logger.error(
                f"Can not find iteration for embedding model {task.embedding_model_id}"
            )
            return

        # Construct model path for query type
        query_model_storage_info = ModelStorageInfo(
            model_repo=model_repo,
            deployed_model_info=DeployedModelInfo(
                plugin_name=iteration.plugin_name,
                model_type="query",
                embedding_model_id=task.embedding_model_id,
                version="1",
            ),
        )

        # Check if this query model is used as same_query (shared with items)
        same_query = os.path.exists(
            os.path.join(query_model_storage_info.model_path, "same_query")
        )

        # Delete query model directory
        shutil.rmtree(query_model_storage_info.model_path)

        # If model is not shared with "items", delete the "items" model path
        if not same_query:
            items_model_storage_info = ModelStorageInfo(
                model_repo=model_repo,
                deployed_model_info=DeployedModelInfo(
                    plugin_name=iteration.plugin_name,
                    model_type="items",
                    embedding_model_id=task.embedding_model_id,
                    version="1",
                ),
            )
            shutil.rmtree(items_model_storage_info.model_path)

        # Finalize task
        task.status = TaskStatus.done
        context.model_deletion_task.update(obj=task)

    except Exception:
        # Catch all exceptions and mark task as failed
        logger.exception(
            f"Something went wrong during model deletion with ID: {model_id}"
        )
        task.status = TaskStatus.failed
        context.model_deletion_task.update(obj=task)

    finally:
        # Always release the lock
        release_lock(lock_file)
