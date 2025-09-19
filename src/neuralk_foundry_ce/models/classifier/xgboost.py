import numpy as np
from .base import ClassifierModel
from ...utils.splitting import with_masked_split
from ...config import global_config


class XGBoostClassifier(ClassifierModel):
    """
    Train an XGBoost classifier on tabular data.

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
    Standard XGBoost hyperparameters can be passed to control training behavior,
    such as `n_estimators`, `learning_rate`, `max_depth`, etc.

    Notes
    -----
    Requires `xgboost` to be installed.
    """
    name = "xgboost-classifier"

    def __init__(self):
        super().__init__()

    def init_model(self, config):
        import xgboost as xgb

        self.model = xgb.XGBClassifier(**config)
        self.config = config

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return self.model.predict(X)

    def get_fixed_params(self, inputs):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            #"scale_pos_weight": 0.97 / 0.03,
        }
        if any(inputs['X'].dtypes == 'category'):
            params['enable_categorical'] = True

        if np.unique(inputs['y']).shape[0] >= 3:
            params['objective'] = 'multi:softprob'

        params['device'] = global_config.device

        return params 

    def get_model_params(self, trial, inputs):
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 500, 1000]),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_categorical("subsample", [0.5, 0.8, 1.0]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 100, log=True),
            "random_state": trial._trial_id
        }