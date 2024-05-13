import gc
import logging
import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional

import torch
from hyperopt import Trials, fmin, hp, tpe

from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.embeddings.data.ranking_data import RankingData
from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.experiments.finetuning_iteration import (
    FineTuningIteration,
)
from embedding_studio.experiments.finetuning_params import FineTuningParams
from embedding_studio.experiments.finetuning_settings import FineTuningSettings
from embedding_studio.workers.fine_tuning.finetune_embedding_one_param import (
    fine_tune_embedding_model_one_param,
)
from embedding_studio.workers.fine_tuning.worker_exceptions import (
    BestParamsNotFoundError,
    ModelNotFoundError,
    ParamsNotFoundError,
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
        logger.error(
            f"Failed hyperopt run with exception: {str(e)}\nTraceback:\t{traceback.format_exc()}"
        )

    return quality if tracker.is_loss else -1 * quality


def finetune_embedding_model(
    iteration: FineTuningIteration,
    settings: FineTuningSettings,
    ranking_data: RankingData,
    query_retriever: QueryRetriever,
    tracker: ExperimentsManager,
    initial_params: Dict[str, List[Any]],
    initial_max_evals: int = 100,
):
    """Start embedding fine-tuning iteration.

    :param iteration: fine-tuning iteration info
    :param settings: fine-tuning settings
    :param ranking_data: dataset with clickstream and items
    :param query_retriever: object to get item related to query, that can be used in "forward"
    :param tracker: experiment management object
    :param initial_params: initial huperparams
    :param initial_max_evals: max initial hyperparams (default: 100)
    :return:
    """
    if not isinstance(initial_max_evals, int) or initial_max_evals <= 0:
        raise ValueError("initial_max_evals should be a positive integer")

    if len(initial_params) == 0:
        raise ValueError("initial_params should not be empty")

    tracker.set_iteration(iteration)
    logger.info("Start fine-tuning iteration")

    best_params = None
    if not tracker.is_initial_run(iteration.run_id):
        starting_run_param: Optional[
            FineTuningParams
        ] = tracker.get_params_by_run_id(iteration.run_id)
        if starting_run_param is None:
            logger.error(
                f"Cannot get fine-tuning params for starting run with ID {iteration.run_id}."
            )
            raise ParamsNotFoundError(iteration.run_id)

        starting_run_experiment_id: str = tracker.get_experiment_id(
            iteration.run_id
        )
        best_params: Optional[
            List[FineTuningParams]
        ] = tracker.get_top_params_by_experiment_id(starting_run_experiment_id)
        if best_params is None:
            logger.error(
                f"Cannot get fine-tuning params for starting experiment with ID: {starting_run_experiment_id}"
            )
            raise BestParamsNotFoundError(starting_run_experiment_id)

        best_params = [starting_run_param] + best_params

    with tempfile.TemporaryDirectory() as tmpdirname:
        initial_model_path: str = os.path.join(tmpdirname, "initial_model.pth")
        try:
            initial_model: EmbeddingsModelInterface = (
                tracker.download_model_by_run_id(iteration.run_id)
            )
            if initial_model is None:
                logger.error(
                    f"Cannot find a model with run ID: {iteration.run_id}"
                )
                raise ModelNotFoundError(iteration.run_id)

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
                _ = fmin(
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
                    verbose=False,
                )

            else:
                logger.info(
                    f"Use {len(best_params)} best parameters from the previous fine-tuning iteration"
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

                else:
                    tracker.delete_previous_iteration()

        except Exception as e:
            logger.exception(
                f"Iteration is failed due to exception: {str(e)}\nTraceback:\t{traceback.format_exc()}"
            )

        if os.path.exists(initial_model_path):
            os.remove(initial_model_path)

    tracker.finish_iteration()
