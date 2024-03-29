# Hello, Unstructured World!

To try out Embedding Studio, you can launch the pre-configured demonstration project. We've prepared a dataset stored in
a public S3 bucket, an emulator for user clicks, and a basic script for fine-tuning the model. By adapting it to your
requirements, you can initiate fine-tuning for your model.

## Prerequisites

Ensure that you have the `docker compose version` command working on your system:

```bash
Docker Compose version v2.23.3
```

!!! Note

    You can also try the `docker-compose` command. Moving forward, we will use the newer `docker compose`
    command, but the `docker-compose` command may also work successfully on your system.

## Start services

Firstly, bring up all the Embedding Studio services by executing the following command:

```shell
docker compose up -d
```

!!! Warning 

    Embedding Studio is run upon docker-compose v2.17.0 and never, installation manual you can
    find [here](https://docs.docker.com/compose/install/linux/).

Upon building and starting, the following services will be launched:

1. **embedding_studio**: The primary service accessible at `http://localhost:5000`, responsible for the core engine
   functionality.
2. **fine_tuning_worker**: A worker service for model fine-tuning based on user feedback, leveraging NVIDIA GPUs for
   the task.
3. **mlflow**: A service facilitating the tracking of fine-tuning experiments.
4. **mlflow_db**: A MySQL instance for storing MLflow-related data.
5. **mongo**: A MongoDB service for storing user interactions and tasks for model fine-tuning.
6. **redis**: A Redis service for task storage during fine-tuning.
7. **minio**: A MinIO service set up for artifact storage, ensuring a secure location for your data.

## Сlickstream emulation

Once all services are up, you can start using Embedding Studio. Let's simulate a user search session. We'll run a
pre-built script that will invoke the Embedding Studio API and emulate user behavior:

```shell
docker compose --profile demo_stage_clickstream up -d
```

## Fine-tuning

After the script execution, you can initiate model fine-tuning. Execute the following command:

```shell
docker compose --profile demo_stage_finetuning up -d
```

This will queue a task processed by the fine-tuning worker. To fetch all tasks in the fine-tuning queue, send a GET
request to the endpoint `/api/v1/fine-tuning/task`:

```shell
curl -X GET http://localhost:5000/api/v1/fine-tuning/task
```

The answer will be something like:

```json
[
  {
    "fine_tuning_method": "Default Fine Tuning Method",
    "status": "processing",
    "created_at": "2023-12-21T14:30:25.823000",
    "updated_at": "2023-12-21T14:32:16.673000",
    "batch_id": "65844a671089823652b83d43",
    "id": "65844c019fa7cf0957d04758"
  }
]
```

where:

* `fine_tuning_method` - method used for fine-tuning the model. We'll discuss this further later on.
* `status` - status of the task. Possible values: pending, processing, done, canceled, error
* `created_at` - task creation date.
* `updated_at` - last task update date.
* `batch_id` - batch identifier indicating gathered clickstream sessions.
* `id` - task identifier.

Once you have the task ID, you can directly monitor the fine-tuning progress by sending a GET request to the
endpoint `/api/v1/fine-tuning/task/{task_id}`:

```shell
curl -X GET http://localhost:5000/api/v1/fine-tuning/task/65844c019fa7cf0957d04758
```

The result will be similar to what you received when querying all tasks.

## Progress tracking with MLflow

For a more convenient way to track progress, you can use `MLflow` at `http://localhost:5001`. Here, you'll find the
following experiments:

* `Default`: A default experiment generated by MLflow, which we don't use.
* `iteration / initial`: This experiment stores the model used for training, loaded into MLflow using the
  `upload_initial_model` method (see [Plugins](plugins.md)).
* `iteration / Default Fine Tuning Method / 65844a671089823652b83d43`: This experiment is the result of the fine-tuning
  process. Learn more about MLflow in
  [their documentation](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html#step-6-view-the-run-in-the-mlflow-ui).
  Also, you can find more information about the fine-tuning process in the
  section [Fine-tuning tracking](fine_tuning_tracking.md)

!!! Note 
    Fine-tuning is a very long process, so **it can take about 30 minutes (if using a GPU)**.

It's also beneficial to check the logs of the `fine_tuning_worker` to ensure everything is functioning correctly. To do
this, list all services using the command:

```shell
docker ps
```

You'll see output similar to:

```shell
CONTAINER ID   IMAGE                                 COMMAND                  CREATED       STATUS                 PORTS                               NAMES
665eef2e757d   embedding_studio-mlflow               "mlflow server --bac…"   3 hours ago   Up 3 hours             0.0.0.0:5001->5001/tcp              embedding_studio-mlflow-1
65043da928d4   embedding_studio-fine_tuning_worker   "dramatiq embedding_…"   3 hours ago   Up 3 hours                                                 embedding_studio-fine_tuning_worker-1
c930d9ca07c0   embedding_studio-embedding_studio     "uvicorn embedding_s…"   3 hours ago   Up 3 hours (healthy)   0.0.0.0:5000->5000/tcp              embedding_studio-embedding_studio-1
5e799aaaf17b   redis:6.2-alpine                      "docker-entrypoint.s…"   3 hours ago   Up 3 hours (healthy)   0.0.0.0:6379->6379/tcp              embedding_studio-redis-1
ba608b022828   bitnami/minio:2023                    "/opt/bitnami/script…"   3 hours ago   Up 3 hours (healthy)   0.0.0.0:9000-9001->9000-9001/tcp    embedding_studio-minio-1
914cb70ed622   mysql/mysql-server:5.7.28             "/entrypoint.sh mysq…"   3 hours ago   Up 3 hours (healthy)   0.0.0.0:3306->3306/tcp, 33060/tcp   embedding_studio-mlflow_db-1
493c45f880c0   mongo:4                               "docker-entrypoint.s…"   3 hours ago   Up 3 hours (healthy)   0.0.0.0:27017->27017/tcp            embedding_studio-mongo-1
```

From here, you can access logs for the specific service using its `CONTAINER ID` or `NAME`, e.g., `65043da928d4` or
`embedding_studio-fine_tuning_worker-1`:

```shell
docker logs embedding_studio-fine_tuning_worker-1
```

If everything completes successfully, you'll see logs similar to:

```shell
Epoch 2: 100%|██████████| 13/13 [01:17<00:00,  0.17it/s, v_num=8]
[2023-12-21 14:59:05,931] [PID 7] [Thread-6] [pytorch_lightning.utilities.rank_zero] [INFO] `Trainer.fit` stopped: `max_epochs=3` reached.
Epoch 2: 100%|██████████| 13/13 [01:17<00:00,  0.17it/s, v_num=8]
[2023-12-21 14:59:05,975] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.finetune_embedding_one_param] [INFO] Save model (best only, current quality: 8.426392069685529e-05)
[2023-12-21 14:59:05,975] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.experiments.experiments_tracker] [INFO] Save model for 2 / 9a9509bf1ed7407fb61f8d623035278e
[2023-12-21 14:59:06,009] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.experiments.experiments_tracker] [WARNING] No finished experiments found with model uploaded, except initial
[2023-12-21 14:59:16,432] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.experiments.experiments_tracker] [INFO] Upload is finished
[2023-12-21 14:59:16,433] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.finetune_embedding_one_param] [INFO] Saving is finished
[2023-12-21 14:59:16,433] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.experiments.experiments_tracker] [INFO] Finish current run 2 / 9a9509bf1ed7407fb61f8d623035278e
[2023-12-21 14:59:16,445] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.experiments.experiments_tracker] [INFO] Current run is finished
[2023-12-21 14:59:16,656] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.experiments.experiments_tracker] [INFO] Finish current iteration 2
[2023-12-21 14:59:16,673] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.experiments.experiments_tracker] [INFO] Current iteration is finished
[2023-12-21 14:59:16,673] [PID 7] [Thread-6] [embedding_studio.workers.fine_tuning.worker] [INFO] Fine tuning of the embedding model was completed successfully!
```

**Congratulations! You've successfully improved the model!**