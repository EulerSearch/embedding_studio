# Iteration emulator

In the section [Hello, unstructured World!](hello_unstructured_world.md) there are two simple emulation steps,
just to test that EmbeddingStudio is build and running well. But you also can test EmbeddingStudio on
an emulated dataset to check algorithmic correctness. We have separated these emulations to show how they work.

!!! warning
    By running a full iteration emulator, initial stage of fine-tuning **can take hours**.

## What's the difference with [Hello, unstructured World!](#hello-unstructured-world-)

Section [Hello, unstructured World!](#hello-unstructured-world-) serves a purpose of a simple and quick demo,
just to check that everything is running ok. The very next step is to actually check
whether the service can really improve embedding model. Actual fine-tuning step, especially initial stage is quite long
and can take hours.

## What does `emulation` mean here

It's all simple:

1. We picked the easiest domain and the easiest dataset ([Remote landscapes](https://huggingface.co/datasets/EmbeddingStudio/merged_remote_landscapes_v1)),
   so we definitely can show positive results in the demo;
2. We generated related text queries using GPT3.5;
3. And for each generated text query we emulated search sessions and user clicks (with some probability of a mistake);
4. All data we put into public to read AWS S3 bucket;

More about actual emulation you can 
[find here](https://github.com/EulerSearch/embedding_studio/blob/main/examples/demo/iteration_emulator.py).

## Emulated data

### Dataset

As we mentioned before for the demo we use **the easiest domain and dataset as we can**
- a merged version of following datasets: *torchgeo/ucmerced*, *NWPU-RESISC45*.

!!! note
    This is a union of categories from original datasets:
    
    *agricultural, airplane, airport, baseball diamond, basketball court, beach, bridge, buildings, chaparral, church,
    circular farmland, cloud, commercial area, desert, forest, freeway, golf course, ground track field, harbor, industrial
    area, intersection, island, lake, meadow, mountain, overpass, palace, parking lot, railway, railway station, rectangular
    farmland, residential, river, roundabout, runway, sea ice, ship, snowberg, stadium, storage tanks, tennis court,
    terrace, thermal power station, wetland*.

More information available on our [HuggingFace page](https://huggingface.co/datasets/EmbeddingStudio/merged_remote_landscapes_v1).

!!! warning
    Synonymous and ambiguous categories were combined (see "Merge method").

For being easily used for the demo we put all items of this dataset into public for reading AWS S3 Bucket:

* **Region name**: us-west-2
* **Bucket name**: embedding-studio-experiments
* **Path to items**: remote-lanscapes/items/

### Clickstream

We pre-generated a batch of clickstream sessions. To check the algorithm of generation,
please visit our [experiments repo](https://github.com/EulerSearch/embedding_studio_experiments/blob/main).

Briefly, the generation method is:

1. For each category were generated up to 20 queries using GPT-3.5.
2. Using VIT-B-32 OpenAI CLIP, and Faiss.FlatIndexIP for each query were emulated search sessions.
3. And then for each search session out of each positive (related to a category of a given query) we pick random set
   as clicks with some probability of a mistake.

Params of emulation:

* A count of search results;
* A range of random picked positives;
* A probability of a mistake;


We put the result of generation into the public reading-available S3 repository:

* Region name: us-west-2
* Bucket name: embedding-studio-experiments
* Path to items: remote-lanscapes/clickstream
* A result of generation with different conditions were packed into different folders
* Generation params are available by path: remote-lanscapes/clickstream/{generation-id}/conditions.json
* Generated clickstreams are available by path: remote-lanscapes/clickstream/{generation-id}/sessions.json

## How to start

Once you've started EmbeddingStudio locally by executing:
```shell
docker compose up -d 
```

To run full iteration you can execute following command:
```shell
docker compose --profile demo_stage_full_iteration up -d
```

It's also beneficial to check the logs of the `fine_tuning_worker` to ensure everything is functioning correctly. To do
this, list all services using the command:

```shell
docker ps
```

You'll see output similar to:
```shell
CONTAINER ID   IMAGE                                         COMMAND                  CREATED          STATUS                 PORTS                               NAMES
ad3a8321e637   embedding_studio-iteration_emulator           "python demo/iterati…"   25 seconds ago   Up 1 second                                                embedding_studio-iteration_emulator-1
665eef2e757d   embedding_studio-mlflow                       "mlflow server --bac…"   3 hours ago      Up 3 hours             0.0.0.0:5001->5001/tcp              embedding_studio-mlflow-1
65043da928d4   embedding_studio-fine_tuning_worker           "dramatiq embedding_…"   3 hours ago      Up 3 hours                                                 embedding_studio-fine_tuning_worker-1
c930d9ca07c0   embedding_studio-embedding_studio             "uvicorn embedding_s…"   3 hours ago      Up 3 hours (healthy)   0.0.0.0:5000->5000/tcp              embedding_studio-embedding_studio-1
5e799aaaf17b   redis:6.2-alpine                              "docker-entrypoint.s…"   3 hours ago      Up 3 hours (healthy)   0.0.0.0:6379->6379/tcp              embedding_studio-redis-1
ba608b022828   bitnami/minio:2023                            "/opt/bitnami/script…"   3 hours ago      Up 3 hours (healthy)   0.0.0.0:9000-9001->9000-9001/tcp    embedding_studio-minio-1
914cb70ed622   mysql/mysql-server:5.7.28                     "/entrypoint.sh mysq…"   3 hours ago      Up 3 hours (healthy)   0.0.0.0:3306->3306/tcp, 33060/tcp   embedding_studio-mlflow_db-1
493c45f880c0   mongo:4                                       "docker-entrypoint.s…"   3 hours ago      Up 3 hours (healthy)   0.0.0.0:27017->27017/tcp            embedding_studio-mongo-1
```

From here, you can access logs for the specific service using its `CONTAINER ID` or `NAME`, e.g., `65043da928d4` or
`embedding_studio-fine_tuning_worker-1`, for details check [here](hello_unstructured_world.md#progress-tracking-with-mlflow).

You can check emulator log by executing:
```shell
docker logs embedding_studio-iteration_emulator-1
```

If everything completes successfully, you'll see logs similar to:

```shell
Download emulated clickstream sessions from S3 Bucket: embedding-studio-experiments by path remote-lanscapes/clickstream/f6816566-cac3-46ac-b5e4-0d5b76757c93/sessions.json
No specific AWS credentials, use Anonymous session
Downloaded 683 emulated clickstream sessions
Use 600 of 683 for emulation
100%|██████████| 600/600 [00:05<00:00, 119.88it/s]
```

Once the process is finished, you can tell the best model. See [How to get best model?](how_to_get_best_model.md)