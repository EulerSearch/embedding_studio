# Plugins

Embedding Studio supports plugins for fine-tuning models. A plugin is a script that inherits from the
[`FineTuningMethod`](https://github.com/EulerSearch/embedding_studio/blob/v0.0.1/embedding_studio/core/plugin.py#L10) 
class and implements the `upload_initial_model` and `get_fine_tuning_builder` methods. Plugins can be of any type; you 
can use any libraries and frameworks for model fine-tuning.

The path to the plugins directory is specified in the `ES_PLUGINS_PATH` environment variable. By default, it points
to the `plugins` directory at the project's root. You can easily change this in the `.env` file.

We provide a demonstration plugin named 
[`Default Fine Tuning Method`](https://github.com/EulerSearch/embedding_studio/blob/v0.0.1/plugins/default_fine_tuning_method.py#L55).

Let's dive into how it works:

```python
class DefaultFineTuningMethod(FineTuningMethod):
    meta = PluginMeta(
        name="Default Fine Tuning Method",
        version="0.0.1",
        description="A default fine-tuning plugin",
    )
    ...
```

The class name can be arbitrary, but it must inherit from `FineTuningMethod`. In the `meta` field, you specify metadata
about the plugin. This is used by Embedding Studio to determine which plugin to use for fine-tuning. The `meta.name`
field is essential because it's used to create tasks for `fine_tuning_worker`.

Next, let's look at the `upload_initial_model` method:

```python
def upload_initial_model(self) -> None:
    model = TextToImageCLIPModel(SentenceTransformer("clip-ViT-B-32"))
    self.manager.upload_initial_model(model)
```

In this function, we define **the initial model for fine-tuning**. In our case, it's the `TextToImageCLIPModel` model
composed of the `SentenceTransformer` model named `clip-ViT-B-32`. We upload it to Mlflow for future use in fine-tuning.
The call to `self.manager.upload_initial_model(model)` is mandatory.

Now, let's examine the plugin **initialization method**. We've tried to describe what each line does in the comments:

```python
def __init__(self):
    # uncomment and pass your credentials to use your own s3 bucket
    # creds = {
    #     "role_arn": "arn:aws:iam::123456789012:role/some_data"
    #     "aws_access_key_id": "TESTACCESSKEIDTEST11",
    #     "aws_secret_access_key": "QWERTY1232qdsadfasfg5349BBdf30ekp23odk03",
    # }
    # self.data_loader = AwsS3DataLoader(**creds)

    # with empty creds, use anonymous session
    creds = {
    }
    self.data_loader = AWSS3DataLoader(**creds)

    self.retriever = TextQueryRetriever()
    self.parser = AWSS3ClickstreamParser(
        TextQueryItem, SearchResult, DummyEventType
    )
    self.splitter = ClickstreamSessionsSplitter()
    self.normalizer = DatasetFieldsNormalizer("item", "item_id")
    self.storage_producer = CLIPItemStorageProducer(self.normalizer)

    self.accumulators = [
        MetricsAccumulator("train_loss", True, True, True, True),
        MetricsAccumulator(
            "train_not_irrelevant_dist_shift", True, True, True, True
        ),
        MetricsAccumulator(
            "train_irrelevant_dist_shift", True, True, True, True
        ),
        MetricsAccumulator("test_loss"),
        MetricsAccumulator("test_not_irrelevant_dist_shift"),
        MetricsAccumulator("test_irrelevant_dist_shift"),
    ]

    self.manager = ExperimentsManager(
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        main_metric="test_not_irrelevant_dist_shift",
        accumulators=self.accumulators,
    )

    self.initial_params = INITIAL_PARAMS
    self.initial_params.update(
        {
            "not_irrelevant_only": [True],
            "negative_downsampling": [
                0.5,
            ],
            "examples_order": [
                [
                    11,
                ]
            ],
        }
    )

    self.settings = FineTuningSettings(
        loss_func=CosineProbMarginRankingLoss(),
        step_size=35,
        test_each_n_sessions=0.5,
        num_epochs=3,
    )
```

Finally, let's look at the `get_fine_tuning_builder` method:

```python
def get_fine_tuning_builder(
        self, clickstream: List[SessionWithEvents]
) -> FineTuningBuilder:
    ranking_dataset = prepare_data(
        clickstream,
        self.parser,
        self.splitter,
        self.retriever,
        self.data_loader,
        self.storage_producer,
    )
    fine_tuning_builder = FineTuningBuilder(
        data_loader=self.data_loader,
        query_retriever=self.retriever,
        clickstream_parser=self.parser,
        clickstream_sessions_splitter=self.splitter,
        dataset_fields_normalizer=self.normalizer,
        item_storage_producer=self.storage_producer,
        accumulators=self.accumulators,
        experiments_manager=self.manager,
        fine_tuning_settings=self.settings,
        initial_params=self.initial_params,
        ranking_data=ranking_dataset,
        initial_max_evals=2,
    )
    return fine_tuning_builder
```

In this method, we describe **how the model fine-tuning will take place**. In our case, we use the
[`prepare_data`](https://github.com/EulerSearch/embedding_studio/blob/v0.0.1/embedding_studio/workers/fine_tuning/data/prepare_data.py#L32) 
function to transform the clickstream into a dataset suitable for fine-tuning. Then, we create an instance of the 
`FineTuningBuilder` class, which will perform the fine-tuning. In the constructor, we pass all the necessary components 
that will be used during the fine-tuning process.
