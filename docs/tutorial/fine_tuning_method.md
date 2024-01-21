# Fine-Tuning Method

This is the main part you'll need to implement yourself.

You need to write a script that initiates the fine-tuning of
the model. This script inherits from
the [`FineTuningMethod`](https://github.com/EulerSearch/embedding_studio/blob/v0.0.1/embedding_studio/core/plugin.py#L10)
class and should implement methods:

* `upload_initial_model`: used for uploading your model to MLflow
* `get_fine_tuning_builder`: used for configuring the model's fine-tuning.

An example of such a script can be found in the
directory [plugins/default_fine_tuning_method.py](https://github.com/EulerSearch/embedding_studio/blob/main/plugins/default_fine_tuning_method.py#L55).

You can have multiple fine-tuning scripts. The choice of which script to use occurs when launching the task:

```shell
curl -X POST http://localhost:5000/api/v1/fine-tuning/task \
  -H 'Content-Type: application/json' \
  -d '{
    "fine_tuning_method": "Default Fine Tuning Method"
}'
```
where `fine_tuning_method` is the name of your fine-tuning script, taken from the `meta.name` field.

!!! note
    The `meta.name` field is a required field in your `FineTuningMethod`.

The path to the plugins directory is specified in the `ES_PLUGINS_PATH` environment variable. By default, it points to
the `plugins` directory at the project's root. It's easiest to change this in the `.env` file.

For more details on plugins, see the [Plugins](plugins.md) section.
