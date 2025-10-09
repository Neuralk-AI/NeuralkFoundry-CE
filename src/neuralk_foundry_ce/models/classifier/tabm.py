from .base import ClassifierModel
from ...utils.splitting import with_masked_split
import pandas as pd
import numpy as np

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
        self.tunable = True
        self.n_ensemble = 50
        
    def init_model(self, config):
        self.config = config

    @with_masked_split
    def train(self, X, y):
        from autogluon.tabular import TabularPredictor

        self.label = "target"
        self.model = TabularPredictor(label =self.label)
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        df_train = X.copy()
        df_train[self.label] = y
        hyperparams = {"TABM": {},
                        }
    
        self.model.fit(
            train_data = df_train,
            hyperparameters = hyperparams,
        )

    @with_masked_split
    def forward(self, X):
        X = X.reset_index(drop=True)
        self.extras['y_score'] = np.array(self.model.predict_proba(X))
        return np.array(self.model.predict(X))

    def get_fixed_params(self, tags):
        return {
            "verbosity": 0,
         }
    
    def get_model_params(self, trial, tags): # According to TabArena Hyperparameter search space
        return {
            "lr": trial.suggest_float("lr", 0.0001, 0.003, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "d_embedding": trial.suggest_int("d_embedding", 8, 32, step=4),
        }
