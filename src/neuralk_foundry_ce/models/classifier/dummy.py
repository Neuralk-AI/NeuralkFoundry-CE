import numpy as np
import cupy as cp
from .base import ClassifierModel
from ...utils.splitting import with_masked_split
from ...config import global_config

from sklearn.dummy import DummyClassifier as _DummyClassifier


class DummyClassifier(ClassifierModel):
    """
    Train a Dummy Classifier

    Inputs
    ------
    - X : Feature matrix for training or prediction.
    - y : Target labels (for training only).
    - splits : Optional train/val/test split masks.

    Outputs
    -------
    - y_pred : Predicted class labels.
    - y_score : Class probabilities (if available and requested).
    """
    name = "dummy"

    def __init__(self):
        super().__init__()

    def init_model(self, config):
        self.config = config
        self.model = _DummyClassifier()

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return self.model.predict(X)

    def get_fixed_params(self, inputs):
        return { }
    
    def get_model_params(self, trial, inputs):
        return { }

