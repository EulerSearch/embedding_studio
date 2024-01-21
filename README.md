<p align="center">
  <img src="docs/images/embedding_studio_logo.svg" alt="EmbeddingStudio" />
</p>

<p align="center">
    <a href="#"><img src="https://img.shields.io/badge/version-0.0.1-orange.svg" alt="version"></a>
    <a href="https://www.python.org/downloads/release/python-3918/"><img src="https://img.shields.io/badge/python-3.9-blue.svg" alt="Python 3.9"></a>
    <a href="#"><img src="https://img.shields.io/badge/CUDA-11.7.1-green.svg" alt="CUDA 11.7.1"></a>
    <a href="#"><img src="https://img.shields.io/badge/docker--compose-2.17.0-blue.svg" alt="Docker Compose Version"></a>
</p>

<p align="center">
    <a href="https://embeddingstud.io/tutorial/hello_unstructured_world/">Tutorial</a> •
    <a href="https://embeddingstud.io/tutorial/getting_started/">Documentation</a> •    
    <a href="https://embeddingstud.io/overview/">Overview</a>
</p>

**EmbeddingStudio** is an innovative open-source framework designed to seamlessly convert a combined
embedding model and vector database into a comprehensive search engine. With built-in functionalities for
clickstream collection, continuous improvement of search experiences, and automatic adaptation of
the embedding model, it offers an out-of-the-box solution for a full-cycle search engine.

## Features

1. 🔄 Turn your vector database into a full-cycle search engine
2. 🖱️ Collect users feedback like clickstream
3. 🚀 (*) Improve search experience on-the-fly without frustrating wait times
4. 📊 (*) Monitor your search quality
5. 🎯 Improve your embedding model through an iterative metric fine-tuning procedure
6. 🆕 (*) Use the new version of the embedding model for inference

(*) - features in development

EmbeddingStudio is highly customizable, so you can bring your own:

1. Data source
2. Vector database
3. Clickstream database
4. Embedding model

## Overview

Our framework enables you to continuously fine-tune your model based on user experience, allowing you to form search 
results for user queries faster and more accurately.

$\color{red}{\textsf{RED:}}$ On the graph, typical search solutions without enhancements, 
such as Full Text Searching (FTS), Nearest Neighbor Search (NNS), and others, are marked in red. Without the use of 
additional tools, the search quality remains unchanged over time.

$\color{orange}{\textsf{ORANGE:}}$ Solutions are depicted that accumulate some feedback (clicks, reviews, votes, discussions, etc.) and then
initiate a full model retraining. The primary issue with these solutions is that full model retraining is a
time-consuming and expensive procedure, thus lacking reactive adjustments (for example, when a product suddenly
experiences increased demand, and the search system has not yet adapted to it).

$\color{#6666ff}{\textsf{INDIGO:}}$ We propose a solution that allows collecting user feedback and rapidly retraining the model on the difference between
the old and new versions. This enables a smoother and more relevant search quality curve for your system.

![Embedding Studio Chart](assets/embedding_studio_chart.png)

## Documentation

View our [official documentation](https://embeddingstud.io/tutorial/getting_started/).

## Getting Started

### Hello, Unstructured World!

To try out EmbeddingStudio, you can launch the pre-configured demonstration project. We've prepared a dataset stored in
a public S3 bucket, an emulator for user clicks, and a basic script for fine-tuning the model. By adapting it to your
requirements, you can initiate fine-tuning for your model.


Ensure that you have the `docker compose version` command working on your system:
```bash
Docker Compose version v2.23.3
```
You can also try the docker-compose version command. Moving forward, we will use the newer docker compose version command, 
but the docker-compose version command may also work successfully on your system.

Firstly, bring up all the EmbeddingStudio services by executing the following command:

```shell
docker compose up -d
```

Once all services are up, you can start using EmbeddingStudio. Let's simulate a user search session. We'll run a
pre-built script that will invoke the EmbeddingStudio API and emulate user behavior:

```shell
docker compose --profile demo_stage_clickstream up -d
```

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

Once you have the task ID, you can directly monitor the fine-tuning progress by sending a GET request to the
endpoint `/api/v1/fine-tuning/task/{task_id}`:

```shell
curl -X GET http://localhost:5000/api/v1/fine-tuning/task/65844c019fa7cf0957d04758
```

The result will be similar to what you received when querying all tasks. 
For a more convenient way to track progress, you can use Mlflow at http://localhost:5001.

It's also beneficial to check the logs of the `fine_tuning_worker` to ensure everything is functioning correctly. To do
this, list all services using the command:

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

To download the best model you can use EmbeddingStudio API:
```bash
curl -X GET http://localhost:5000/api/v1/fine-tuning/task/65844c019fa7cf0957d04758
```

If everything is OK, you will see following output:
```json
{
  "fine_tuning_method": "Default Fine Tuning Method", 
  "status": "done", 
  "best_model_url": "http://localhost:5001/get-artifact?path=model%2Fdata%2Fmodel.pth&run_uuid=571304f0c330448aa8cbce831944cfdd", 
  ...
}
```
And `best_model_url` field contains HTTP accessible `model.pth` file.

You can download *.pth file by executing following command:
```bash
wget http://localhost:5001/get-artifact?path=model%2Fdata%2Fmodel.pth&run_uuid=571304f0c330448aa8cbce831944cfdd
```

## Contributing

We welcome contributions to EmbeddingStudio!

## License

EmbeddingStudio is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.