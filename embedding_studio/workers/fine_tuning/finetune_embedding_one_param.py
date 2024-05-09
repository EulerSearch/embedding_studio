import logging
from typing import Optional

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.embeddings.data.ranking_data import RankingData
from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.embeddings.training.embeddings_finetuner import (
    EmbeddingsFineTuner,
)
from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.experiments.finetuning_params import FineTuningParams
from embedding_studio.experiments.finetuning_settings import FineTuningSettings

logger = logging.getLogger(__name__)


class CustomDataCollator:
    def __call__(self, batch):
        return batch


def fine_tune_embedding_model_one_param(
    initial_model: EmbeddingsModelInterface,
    settings: FineTuningSettings,
    ranking_data: RankingData,
    query_retriever: QueryRetriever,
    fine_tuning_params: FineTuningParams,
    tracker: ExperimentsManager,
) -> float:
    """Run embeddings fine-tuning over single fine-tuning params set

    :param initial_model: embedding model itself
    :param settings: fine-tuning settings
    :param ranking_data: dataset with clickstream and items
    :param query_retriever: object to get item related to query, that can be used in "forward"
    :param fine_tuning_params: hyper params of fine-tuning task
    :param tracker: experiment management object
    :return: the best quality value
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )  # TODO: use multiple devices

    if not use_cuda:
        logger.warning("No CUDA is available, use CPU device")

    try:
        # Start run
        is_finished = tracker.set_run(fine_tuning_params)
    except Exception as e:
        logger.exception('Something went wring during run start.')
        return 0.0

    try:
        start_fine_tuning = True
        quality = None
        if is_finished:
            logger.warning(
                f"Run with params {str(fine_tuning_params)} is finished."
            )
            # Read current embedding quality
            quality: Optional[float] = tracker.get_quality()
            _, best_quality = tracker.get_best_quality()
            if quality is not None and best_quality is not None:
                if quality < best_quality:
                    start_fine_tuning = False
                    logger.info(
                        f"Do not retry: Run with params {str(fine_tuning_params)} has not the best quality: {quality} < {best_quality}."
                    )

                elif tracker.model_is_uploaded():
                    start_fine_tuning = False
                    logger.info(
                        f"Do not retry Run with params {str(fine_tuning_params)} has already had a model being uploaded."
                    )
    except Exception as e:
        logger.exception('Something went wring during run checking.')
        tracker.finish_run(as_failed=True)
        return 0.0

    try:
        if start_fine_tuning:
            logger.info("Init embeddings fine-tuner")
            fine_tuner: EmbeddingsFineTuner = EmbeddingsFineTuner.create(
                initial_model,
                settings,
                ranking_data.items,
                query_retriever,
                fine_tuning_params,
                tracker,
            )
            logger.info("Trying to move to the device...")
            fine_tuner.to(device)
            logger.info("Trying to move to the device... OK")
            fine_tuner.preprocess_sessions(ranking_data.clickstream)

            # Init train / test clickstream data loaders
            train_dataloader: DataLoader = DataLoader(
                ranking_data.clickstream["train"],
                batch_size=settings.batch_size,
                collate_fn=CustomDataCollator(),
                shuffle=True,
            )
            test_dataloader: DataLoader = DataLoader(
                ranking_data.clickstream["test"],
                batch_size=1,
                collate_fn=CustomDataCollator(),
                shuffle=False,
            )

            # If val loss is not changing - stop training
            early_stop_callback: EarlyStopping = EarlyStopping(
                monitor="val_loss",
                patience=3,
                strict=False,
                verbose=False,
                mode="min",
            )

            logger.info("Start fine-tuning")
            if 0 < settings.test_each_n_sessions <= 1:
                settings.test_each_n_sessions *= len(
                    ranking_data.clickstream["train"]
                )
            # Start fine-tuning
            trainer: Trainer = Trainer(
                max_epochs=settings.num_epochs,
                callbacks=[early_stop_callback],
                val_check_interval=int(
                    settings.test_each_n_sessions
                    if settings.test_each_n_sessions > 0
                    else len(train_dataloader)
                ),
            )
            trainer.fit(fine_tuner, train_dataloader, test_dataloader)

            # Move model back to CPU
            fine_tuner.cpu()

            # Unfix layers
            initial_model.unfix_item_model()
            initial_model.unfix_query_model()

            # Read current embedding quality
            quality: Optional[float] = tracker.get_quality()
            logger.info(f"Save model (best only, current quality: {quality})")
            try:
                # Save model, best only
                tracker.save_model(initial_model, True)
                logger.info("Saving is finished")
            except Exception as e:
                logger.exception(f"Unable to save a model: {str(e)}")

        tracker.finish_run()

        return quality

    except Exception as e:
        logger.exception('Something went wrong during fine-tuning stage.')
        tracker.finish_run(as_failed=True)
        return 0.0

