# Run Fine-Tuning

Fine-tuning is like giving your smart model a bit of extra training on a specific set of examples. Imagine it as a way
to teach your model to become even better at a particular task. In Embedding Studio, we use fine-tuning to make our
search experience sharper and more accurate.

Think of it this way: you've got this smart model that's learned a lot of things (we call it pre-trained), and now you
want it to specialize in something unique to your needs. Fine-tuning helps it become a real pro at that specific job.

Here's an overview of what to expect during the fine-tuning process:

1. **Initial Model Setup**: The process begins by downloading and locally saving either the initial model or the model
   from the most recent iteration.
2. **Hyperparameter Selection**: Next, subsets of hyperparameters are defined for exploration. These may be initial sets
   or derived from the last iteration.
3. **Iteration Creation**: An iteration is created, referred to as an experiment
   in MLFlow terminology.
4. **Hyperparameter Runs**: For each subset of hyperparameters (termed fine-tuning params in our context), a run is
   initiated.
5. **Executing Fine-Tuning**: The fine-tuning run starts, using the specified hyperparameters.
6. **Model Evaluation and Saving**: After each run, the main metric is evaluated. If the result is the best so far, the
   model is saved, and the previously best model is deleted, if feasible.
7. **Iteration Completion**: Upon completing all runs, the previous iteration is removed from the system.
8. **Model Cleanup**: Finally, the locally saved model is deleted.

Once you've gathered enough data, you can initiate the fine-tuning process.

## Create task

To run fine-tuning, send a POST request to the `/api/v1/fine-tuning/task` endpoint:

```shell
curl -X POST http://localhost:5000/api/v1/fine-tuning/task \
  -H 'Content-Type: application/json' \
  -d '{
    "fine_tuning_method": "Default Fine Tuning Method",
    "batch_id":  "65844a671089823652b83d43",
    "metadata": {},
    "idempotency_key": "833fb6b260ff41a88cb75c4280e2e270"
}'
```

where

* `fine_tuning_method` - name of your fine-tuning script, taken from the `meta.name` field.
* `batch_id` - (optional) id of released clickstream session batch. If parameter is not sent,
  current clickstream batch will be released and used automatically.
* `metadata` - (optional) any meta data
* `idempotency_key` - (optional) idempotency key

In response, you will receive task id:

```json
{
  "fine_tuning_method": "Default Fine Tuning Method",
  "status": "processing",
  "created_at": "2023-12-21T14:30:25.823000",
  "updated_at": "2023-12-21T14:32:16.673000",
  "batch_id": "65844a671089823652b83d43",
  "id": "65844c019fa7cf0957d04758"
}
```

## Task status

Using this task ID, you can directly monitor the fine-tuning progress by sending a GET request to the
endpoint `/api/v1/fine-tuning/task/{task_id}`:

```shell
curl -X GET http://localhost:5000/api/v1/fine-tuning/task/65844c019fa7cf0957d04758
```

In response, you will receive:

```json
{
  "fine_tuning_method": "Default Fine Tuning Method",
  "status": "processing",
  "created_at": "2023-12-21T14:30:25.823000",
  "updated_at": "2023-12-21T14:32:16.673000",
  "batch_id": "65844a671089823652b83d43",
  "id": "65844c019fa7cf0957d04758"
}
```

where:

* `fine_tuning_method` - method used for fine-tuning the model. We'll discuss this further later on.
* `status` - status of the task. Possible values: pending, processing, done, canceled, error
* `created_at` - task creation date.
* `updated_at` - last task update date.
* `batch_id` - batch identifier indicating a gathered clickstream sessions.
* `id` - task identifier.

A more convenient way to track status is to use MLflow (more details in 
[Progress tracking with MLflow](hello_unstructured_world.md#progress-tracking-with-mlflow)).

Once the process is finished, you can tell the best model. See [How to get best model?](how_to_get_best_model.md)