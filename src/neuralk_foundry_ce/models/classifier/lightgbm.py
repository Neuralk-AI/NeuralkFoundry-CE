from .base import ClassifierModel
from ...utils.splitting import with_masked_split
import numpy as np
import torch


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

    Notes
    -----
    Requires `lightgbm` to be installed.
    """
    name = "lightgbm-classifier"

    def __init__(self):
        super().__init__()

    def init_model(self, config):
        """
        Initialize the model with given hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments corresponding to model hyperparameters.
            These will be passed directly to the model constructor.
        """
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
            "verbose": -1,
        }
        return params


    def get_model_params(self, trial, inputs):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": trial._trial_id,
        }
