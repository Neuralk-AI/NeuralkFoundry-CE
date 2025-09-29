from .base import ClassifierModel
from ...utils.splitting import with_masked_split
import pandas as pd

class ModernNCAClassifier(ClassifierModel):
    """
    Apply a ModernNCA classifier to tabular data.

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
    Requires `tabrepo` to be installed.
    """  
    name = "modern-nca-classifier"

    def __init__(self):
        super().__init__()
        self.tunable = False

    def init_model(self, config):
        from tabrepo.benchmark.models.ag.modernnca.modernnca_model import ModernNCAModel

        self.config = config
        self.model = ModernNCAModel()


    @with_masked_split
    def train(self, X, y):
        print(self.config)
        self.model.fit(X=X, y=pd.Series(y), **self.config)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return self.model.predict(X)

    def get_fixed_params(self, inputs):
        return {
            "num_cpus": 1, 
            "num_gpus": 1,
            "verbosity": 0,
            "cat_col_names": inputs['categorical_features'],
        }
    
    def get_model_params(self, trial, inputs):
        return {}