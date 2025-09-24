from .base import ClassifierModel
from ...utils.splitting import with_masked_split
import pandas as pd

class TabMClassifier(ClassifierModel):
    """
    Apply a TabM classifier to tabular data.

    Inputs
    ------
    - X : Feature matrix for prediction.
    - y : Target labels for training.
    - splits : Optional train/val/test split masks (handled externally).

    Outputs
    -------
    - y_pred : Predicted class labels.
    - y_score : Class probabilities (stored in `extras`).

    Notes
    -----
    Requires `autogluon` to be installed.
    """  
    name = "tabm-classifier"

    def __init__(self):
        super().__init__()

    def init_model(self, config):
        from autogluon.tabular.models.tabm.tabm_model import TabMModel 

        self.config = config
        self.model = TabMModel()

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X=X, y=pd.Series(y), **self.config)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return self.model.predict(X)

    def get_fixed_params(self, tags):
        return {
            "patience": 16,
            "amp": False,
            "arch_type": "tabm-mini",
            "tabm_k":32,
            "gradient_clipping_norm": 1.0,
            "share_training_batches": False,
         }
    
    def get_model_params(self, trial, tags): # According to TabArena Hyperparameter search space
        return {
            "lr": trial.suggest_float("lr", 0.0001, 0.003, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "d_embedding": trial.suggest_int("d_embedding", 8, 32, step=4),
        }
