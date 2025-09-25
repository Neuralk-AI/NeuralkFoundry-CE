from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


from functools import wraps
import numpy as np

from ..utils.splitting import Split
from ..utils.execution import profile_function
from .hyperopt import hyperoptimize
from ..workflow import Step, Field


class BaseModel(Step):
    """
    Base class for predictive task heads in a machine learning workflow.

    Inputs
    ------
    - X : Input feature matrix.
    - y : Ground-truth target values.
    - splits : Train/validation/test split masks.
    - metric_to_optimize : Metric used for model selection or tuning.

    Outputs
    -------
    - y_pred : Predicted target values.

    Parameters
    ----------
    This is an abstract base class. Subclasses must implement the `forward` method to define
    prediction logic.

    Notes
    -----
    Used as the base class for both classification and regression heads.
    Not intended to be used directly.
    """
    name = 'base-model'
    inputs = [
        Field('X', 'Input features of the dataset'),
        Field('y', 'Target variable to predict'),
        Field('splits', 'Masks for train, validation, and test sets'),
        Field('metric_to_optimize', 'Metric to optimize during model selection or hyperparameter tuning'),
    ]

    outputs = [
        Field('y_pred', 'Predicted target values'),
    ]

    params = [
        Field('n_hyperopt_trials', 'Number of trials attempted for hyperparameter optimization', default=200),
    ]

    def _correct_predicted_proba(self, y_score):
        if len(y_score.shape) == 1:
            # assume binary classification and proba for class 1.
            y_score = np.vstack([1 - y_score, y_score])

        # Sklearn checking that the sum of probabilities equals one is too
        # strict. We add a looser version here.
        y_prob_sum = y_score.sum(axis=1)
        if np.allclose(y_prob_sum, 1, rtol=1e-5):
            # It's close enough, we renormalize to avoid the sklearn warning.
            y_score = y_score / y_prob_sum[0, None]

        return y_score


    def _fit_predict_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_array: Iterable,
    ):
        # Train with profiling
        model, mem_usage, time_usage = profile_function(
            self.train,
            X,
            y=y,
            split_mask=split_array,
            splits=[Split.TRAIN, Split.VAL],
        )

        # Predict on all train/val rows
        y_pred_all = -np.ones(X.shape[0], dtype=int)
        non_none_mask = ~np.isin(split_array, Split.NONE)
        y_pred_all[non_none_mask] = self.forward(
            X,
            split_mask=split_array,
            splits=[Split.TRAIN, Split.VAL, Split.TEST],
        )

        return y_pred_all, mem_usage, time_usage


    def _ensemble_fit_predict_all(self, ensemble, X, y, split_array):
        """
        trials: list of (params, weight)
        model_builder: function(**params) -> model with fit/predict_proba
        Returns weighted average of predict_proba outputs.
        """
        preds = []
        weights = ensemble['weights']
        for params in ensemble['params']:
            self.init_model(params)
            self._fit_predict_all(X, y, split_array)
            preds.append(self.extras['y_score'])
            
        preds = np.stack(preds, axis=0)  # (M, N, C)
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()  # should already be normalized
        y_score = np.tensordot(weights, preds, axes=(0, 0))  # (N, C)
        self.extras['y_score'] = y_score
        return np.argmax(y_score, axis=1)


    def _evaluate(
        self,
        y_pred_all: np.ndarray,
        y: np.ndarray,
        split_array: np.ndarray,
        metrics: Dict[str, Callable[..., float]],
        inputs: Dict[str, Any],
        is_ensemble: bool = False
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train the model with profiling, run inference for all non-NONE splits,
        compute & log test metrics, and publish predictions/scores.

        Returns
        -------
        info : Dict[str, float]
            Profiling info with 'mem_usage' and 'time_usage'.
        """
        suff = 'ensemble_' if is_ensemble else ''

        # Build test slice
        test_mask = np.isin(split_array, [Split.TEST])
        non_none_mask = ~np.isin(split_array, Split.NONE)
        y_test = y[test_mask]
        y_pred = y_pred_all[test_mask]

        test_preds_for_metrics: Dict[str, Any] = {"y_true": y_test, "y_pred": y_pred}

        # Optional: class probabilities
        y_score_all = None
        if "y_score" in getattr(self, "extras", {}):
            y_score = self._correct_predicted_proba(self.extras["y_score"])
            y_score_all = -np.ones((len(split_array), y_score.shape[1]), dtype=y_score.dtype)
            y_score_all[non_none_mask] = y_score
            test_preds_for_metrics["y_score"] = y_score_all[test_mask]

        # Optional: label indices for metrics that require explicit label set
        if "y_classes" in inputs:
            test_preds_for_metrics["labels"] = np.arange(len(inputs["y_classes"]))

        # Compute & log metrics on TEST
        for metric_name, metric_fn in metrics.items():
            value = metric_fn(**test_preds_for_metrics)
            self.log_metric("test_" + suff + metric_name, value)

        # Return artifacts
        if y_score_all is not None:
            self.output(f"y_{suff}score", y_score_all)
        self.output(f"y_{suff}pred", y_pred_all)

        return

    def _execute(self, inputs: dict) -> dict:
        X = inputs['X']
        y = inputs['y']
        splits = inputs["splits"]
        metrics = type(self).get_metrics()
        self.extras = {}
        n_ensemble = getattr(self, 'n_ensemble', None)

        if not hasattr(self, 'tunable') or self.tunable:

            best_trial, ensemble = hyperoptimize(
                {'n_trials': self.n_hyperopt_trials},
                self,
                X, y,
                splits,
                metrics,
                inputs,
                n_ensemble=n_ensemble
            )

            best_params = best_trial.params
            self.init_model(best_params)

            # Take care of ensembles first
            if n_ensemble:
                y_pred_all = self._ensemble_fit_predict_all(ensemble, X, y, splits[0])
                self._evaluate(y_pred_all, y, splits[0], metrics, inputs, is_ensemble=True)

        else:
            self.init_model(self.get_fixed_params(inputs))
            best_trial = None

        y_pred_all, mem_usage, time_usage = self._fit_predict_all(X, y, splits[0])
        self._evaluate(y_pred_all, y, splits[0], metrics, inputs)

        if best_trial is not None:
            self.log_metric("best_hyperopt_params", best_trial.params)
            self.log_metric("best_hyperopt_score", best_trial.value)

        self.log_metric("metric_to_optimize", inputs['metric_to_optimize'])

        # Performance
        self.log_metric("mem_usage", np.max(mem_usage))
        self.log_metric("time_usage", time_usage)

    @abstractmethod
    def forward(self) -> None:
        pass
