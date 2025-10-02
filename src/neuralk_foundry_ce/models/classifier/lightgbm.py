import numpy as np
import torch


from .base import ClassifierModel
from ...utils.splitting import with_masked_split
from ...config import global_config


class LightGBMClassifier(ClassifierModel):
    """
    Train a LightGBM classifier on tabular data.

    Inputs
    ------
    - X : Feature matrix for training or prediction.
    - y : Target labels (for training only).
    - splits : Optional train/val/test split masks.

    Outputs
    -------
    - y_pred : Predicted class labels.
    - y_score : Class probabilities (if available and requested).

    Parameters
    ----------
    Standard LightGBM hyperparameters can be passed to control training behavior,
    such as `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, etc.

    Notes
    -----
    Requires `lightgbm` to be installed.
    """
    name = "lightgbm-classifier"

    def __init__(self):
        super().__init__()
        self.n_ensemble = 50

    def init_model(self, config):
        from lightgbm import LGBMClassifier

        self.model = LGBMClassifier(**config)

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] =  self.model.predict_proba(X)
        return self.model.predict(X)

    def get_fixed_params(self, inputs):

        is_binary = np.unique(inputs['y']).shape[0] == 2
        params = {
            "objective": "binary" if is_binary else "multiclass",
            "metric": "binary_logloss" if is_binary else "multi_logloss",
            "feature_pre_filter": True,
            "min_gain_to_split": 0.1,

            "verbose": -1,
        }
        if global_config.device == 'cuda':
            params['device'] = 'gpu'

        return params


    def get_model_params(self, trial, inputs):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": trial._trial_id,
        }
