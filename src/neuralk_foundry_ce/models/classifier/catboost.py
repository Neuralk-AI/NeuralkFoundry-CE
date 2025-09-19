from .base import ClassifierModel
from ...utils.splitting import with_masked_split
import numpy as np
from ...config import global_config


class CatBoostClassifier(ClassifierModel):
    """
    Train a CatBoost classifier on tabular data.

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
    Standard CatBoost hyperparameters can be passed to control training behavior,
    such as `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, etc.

    Notes
    -----
    Requires `catboost` to be installed.
    """
    name = 'catboost-classifier'

    def __init__(self):
        super().__init__()

    def init_model(self, config):
        from catboost import CatBoostClassifier as _CatBoostClassifier

        self.model = _CatBoostClassifier(verbose=0, **config)

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return np.squeeze(self.model.predict(X))

    def get_fixed_params(self, inputs):
        is_binary = np.unique(inputs['y']).shape[0] == 2
        params = {
            "loss_function": "Logloss" if is_binary else "MultiClass",
            "eval_metric": "Logloss" if is_binary else "MultiClass"
        }
        if 'categorical_features' in inputs:
            params['cat_features'] = inputs['categorical_features']

        if global_config.device == 'cuda':
            params['task_type'] = 'GPU'
        return params


    def get_model_params(self, trial, inputs):
        return {
            "iterations": trial.suggest_categorical("iterations", [100, 500, 1000]),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_seed": trial._trial_id,
        }


