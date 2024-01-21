# Fine-Tuning Worker

The `fine_tuning_worker` serves as a specialized worker responsible for executing tasks related to fine-tuning.
**Typically deployed on machines equipped with GPUs**, these workers enable faster model retraining. It's important to
note
that, currently, our system is compatible **exclusively with Nvidia GPUs**. This ensures optimal performance and
efficiency
in the fine-tuning process.

After writing your plugin ([fine-tuning method](fine_tuning_method.md)), you can start the EmbeddingStudio
worker - `fine_tuning_worker`.
To do this, you need to build an image with your plugin and start the worker:

Rebuild the `fine_tuning_worker` image with your plugin and start it:

```shell
docker compose build --no-cache fine_tuning_worker
```

and

```shell
docker compose up -d fine_tuning_worker
```

It will pick up your plugin and wait for fine-tuning tasks.

!!! tip

    To avoid rebuilding the image every time your change it, you can mount the `plugins` directory inside the container. 
    To do this, add a `volume` section to the `docker-compose.yml`:

    ```yaml
    services:
      ...
      fine_tuning_worker:
        ...
        volumes:
          - ./plugins:/embedding_studio/plugins
      ...
    ```

    and start the container:

    ```shell
    docker compose up -d fine_tuning_worker
    ```
