from .base import ClassifierModel
from ...utils.splitting import with_masked_split
from ...config import global_config


class TabICLClassifier(ClassifierModel):
    """
    Apply a TabICL classifier to tabular data.

    Inputs
    ------
    - X : Feature matrix for prediction.
    - y : Target labels for training.
    - splits : Optional train/val/test split masks (handled externally).

    Outputs
    -------
    - y_pred : Predicted class labels.
    - y_score : Class probabilities (stored in `extras`).
    """  
    name = 'tabicl-classifier'

    def __init__(self):
        super().__init__()
        self.tunable = False

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return self.model.predict(X)

    def init_model(self, config):
        from tabicl import TabICLClassifier
        config['device'] = global_config.device

        self.model = TabICLClassifier(**config)
        self.config = config

    def get_fixed_params(self, inputs):
        return {}

    def get_model_params(self, trial, tags):
        return {}