from abc import abstractmethod
import numpy as np

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
        Field('n_hyperopt_trials', 'Number of trials attempted for hyperparameter optimization', default=10),
    ]

    def _execute(self, inputs: dict) -> dict:
        X = inputs['X']
        y = inputs['y']
        splits = inputs["splits"]
        metrics = type(self).get_metrics()
        self.extras = {}

        if not hasattr(self, 'tunable') or self.tunable:

            best_trial = hyperoptimize(
                {'n_trials': self.n_hyperopt_trials},
                self,
                X, y,
                splits,
                metrics,
                inputs,
            )

            best_params = best_trial.params | self.get_fixed_params(inputs)
            self.init_model(best_params)
        else:
            self.init_model(self.get_fixed_params(inputs))
            best_trial = None

        model, mem_usage, time_usage = profile_function(self.train, X, y=y, split_mask=splits[0], splits=[Split.TRAIN, Split.VAL])
        y_pred_all = -np.ones(X.shape[0], dtype=int)
        y_pred_all[~np.isin(np.array(splits[0]), Split.NONE)] = self.forward(
            X, split_mask=splits[0], splits=[Split.TRAIN, Split.VAL, Split.TEST])

        # Compute metrics on test
        test_mask = np.isin(np.array(splits[0]), [Split.TEST])
        y_test, y_pred = y[test_mask], y_pred_all[test_mask]

        test_preds_for_metrics = {'y_pred': y_pred, 'y_true': y_test}

        train_mask = np.isin(np.array(splits[0]), [Split.TRAIN])
        y_train, y_train_pred = y[train_mask], y_pred_all[train_mask]
  
        if 'y_score' in self.extras:
            y_score = self.extras['y_score']
            y_score_all = -np.ones((X.shape[0], y_score.shape[1]), dtype=y_score.dtype)
            y_score_all[~np.isin(np.array(splits[0]), Split.NONE)] = y_score
            
            # Sklearn checking that the sum of probabilities equals one is too
            # strict. We add a looser version here.

            y_prob_sum = y_score_all.sum(axis=1)
            if np.allclose(y_prob_sum, 1, rtol=1e-5):
                # It's close enough, we renormalize to avoid the sklearn warning.
                y_score_all = y_score_all / y_prob_sum[0, None]

            test_preds_for_metrics['y_score'] = y_score_all[test_mask]
        if 'y_classes' in inputs:
            test_preds_for_metrics['labels'] = np.arange(len(inputs['y_classes']))
        for metric_name, metric in metrics.items():
            value = metric(**test_preds_for_metrics)
            self.log_metric('test_' + metric_name, value)

        if 'y_score' in self.extras:
            self.output('y_score', y_score_all)
        self.output('y_pred', y_pred_all)

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
