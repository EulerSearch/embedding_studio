import gc
import logging
import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional

import torch
from hyperopt import Trials, fmin, hp, tpe

from embedding_studio.embeddings.data.clickstream.query_retriever import (
    QueryRetriever,
)
from embedding_studio.embeddings.data.ranking_data import RankingData
from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.worker.experiments.experiments_tracker import (
    ExperimentsManager,
)
from embedding_studio.worker.experiments.finetuning_params import (
    FineTuningParams,
)
from embedding_studio.worker.experiments.finetuning_session import (
    FineTuningSession,
)
from embedding_studio.worker.experiments.finetuning_settings import (
    FineTuningSettings,
)
from embedding_studio.worker.finetune_embedding_one_param import (
    fine_tune_embedding_model_one_param,
)

logger = logging.getLogger(__name__)


def _finetune_embedding_model_one_step(
    initial_model_path: str,
    settings: FineTuningSettings,
    ranking_data: RankingData,
    query_retriever: QueryRetriever,
    fine_tuning_params: FineTuningParams,
    tracker: ExperimentsManager,
):
    logger.debug(f"Read model from local path: {initial_model_path}")
    model: EmbeddingsModelInterface = torch.load(initial_model_path)
    quality: float = fine_tune_embedding_model_one_param(
        model,
        settings,
        ranking_data,
        query_retriever,
        fine_tuning_params,
        tracker,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return quality


def _finetune_embedding_model_one_step_hyperopt(
    initial_model_path: str,
    settings: FineTuningSettings,
    ranking_data: RankingData,
    query_retriever: QueryRetriever,
    hyperopt_params: dict,
    tracker: ExperimentsManager,
) -> float:
    quality = 0.0
    try:
        quality: float = _finetune_embedding_model_one_step(
            initial_model_path,
            settings,
            ranking_data,
            query_retriever,
            FineTuningParams(**hyperopt_params),
            tracker,
        )
    except Exception as e:
        logger.error(f"Failed hyperopt run with exception: {str(e)}")
        logger.debug(f"Traceback:\t{traceback.format_exc()}")

    return quality if tracker.is_loss else -1 * quality


def finetune_embedding_model(
    session: FineTuningSession,
    settings: FineTuningSettings,
    ranking_data: RankingData,
    query_retriever: QueryRetriever,
    tracker: ExperimentsManager,
    initial_params: Dict[str, List[Any]],
    initial_max_evals: int = 100,
):
    """Start embedding fine-tuning session.

    :param session: fine-tuning session info
    :type FineTuningSession
    :param settings: fine-tuning settings
    :type settings: FineTuningSettings
    :param ranking_data: dataset with clickstream and items
    :type ranking_data: RankingData
    :param query_retriever: object to get item related to query, that can be used in "forward"
    :type query_retriever: QueryRetriever
    :param fine_tuning_params: hyper params of fine-tuning task
    :type fine_tuning_params: FineTuningParams
    :param tracker: experiment management object
    :type tracker: ExperimentsManager
    :param initial_params: initial huperparams
    :type initial_params: Dict[str, List]
    :param initial_max_evals: max initial hyperparams (default: 100)
    :type initial_max_evals: int
    :return:
    """
    logger.info("Start fine-tuning session")
    if not isinstance(initial_max_evals, int) or initial_max_evals <= 0:
        raise ValueError("initial_max_evals should be a positive integer")

    if len(initial_params) == 0:
        raise ValueError("initial_params should not be empty")

    best_params: Optional[List[FineTuningParams]] = tracker.get_top_params()
    tracker.set_session(session)
    with tempfile.TemporaryDirectory() as tmpdirname:
        initial_model_path: str = os.path.join(tmpdirname, "initial_model.pth")
        try:
            initial_model: EmbeddingsModelInterface = tracker.get_last_model()
            logger.info(f"Save model to {initial_model_path}")
            torch.save(initial_model, initial_model_path)
            del initial_model
            gc.collect()
            torch.cuda.empty_cache()

            if not best_params:
                logger.info(
                    "Looks like this is the initial run, so hyperopt will be run over provided initial_params"
                )
                initial_hyper_params: Dict[str, Any] = dict()
                for key, value in initial_params.items():
                    initial_hyper_params[key] = hp.choice(key, value)

                trials = Trials()
                logger.info(
                    f"Start hyper parameters optimization process (max evals: {initial_max_evals})"
                )
                best = fmin(
                    lambda params: _finetune_embedding_model_one_step_hyperopt(
                        initial_model_path,
                        settings,
                        ranking_data,
                        query_retriever,
                        params,
                        tracker,
                    ),
                    initial_hyper_params,
                    algo=tpe.suggest,
                    max_evals=initial_max_evals,
                    trials=trials,
                    verbose=1,
                )

            else:
                logger.info(
                    f"Use {len(best_params)} best parameters from the previous fine-tuning session"
                )
                failed_runs_count = 0
                for index, finetuning_params in enumerate(best_params):
                    logger.info(f"Start {index + 1} / {len(best_params)} run")
                    try:
                        _finetune_embedding_model_one_step(
                            initial_model_path,
                            settings,
                            ranking_data,
                            query_retriever,
                            finetuning_params,
                            tracker,
                        )
                        logger.info(
                            f"Finish {index + 1} / {len(best_params)} run"
                        )
                    except Exception as e:
                        logger.exception(
                            f"Failed {index + 1} / {len(best_params)} run with exception: {str(e)}\nTraceback:\t{traceback.format_exc()}"
                        )
                        failed_runs_count += 1

                if failed_runs_count == len(best_params):
                    logger.error(f"Something went wrong, all runs were failed")

        except Exception as e:
            logger.exception(
                f"Session is failed due to exception: {str(e)}\nTraceback:\t{traceback.format_exc()}"
            )

        if os.path.exists(initial_model_path):
            os.remove(initial_model_path)

    tracker.finish_session()
