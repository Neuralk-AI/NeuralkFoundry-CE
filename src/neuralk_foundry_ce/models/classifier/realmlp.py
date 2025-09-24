from .base import ClassifierModel
from ...utils.splitting import with_masked_split
import pandas as pd

class RealMLPClassifier(ClassifierModel):
    """
    Apply a RealMLP classifier to tabular data.

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
    name = "realmlp-s-classifier"

    def __init__(self):
        super().__init__()
        self.tunable = False

    def init_model(self, config):
        from autogluon.tabular.models.realmlp.realmlp_model import RealMLPModel 

        self.config = config
        self.model = RealMLPModel(hyperparameters={"default_hyperparameters": "td_s"}, **config)

    @with_masked_split
    def train(self, X, y):
        
        self.model.fit(X=X, y=pd.Series(y))

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return self.model.predict(X)

    def get_fixed_params(self, tags):
        return { }
    
    def get_model_params(self, trial, tags):
        return { }
