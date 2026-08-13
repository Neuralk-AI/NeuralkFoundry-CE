import numpy as np

from .base import ClassifierModel
from ...utils.splitting import with_masked_split


class SeldonClassifier(ClassifierModel):
    """Neuralk's hosted Seldon (formerly NICL) classifier over HTTP.

    Reads the API key from the NEURALK_API_KEY env var. Requires the
    ``neuralk`` package (not ``neuralk_model``, which is the local variant).
    """
    name = 'seldon-classifier'

    def __init__(self, model: str = 'nicl-small', timeout_s: int = 900):
        super().__init__()
        self.tunable = False
        self.seldon_model = model
        self.timeout_s = timeout_s

    def init_model(self, config):
        # No local backbone to build. Client is instantiated on train() so
        # each fold uses its own client (the API is stateful w.r.t. the
        # context passed to fit).
        self.config = config

    @with_masked_split
    def train(self, X, y):
        from neuralk import SeldonClassifier as _SeldonAPIClient
        self.model = _SeldonAPIClient(model=self.seldon_model, timeout_s=self.timeout_s)
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y).ravel())
        self.classes_ = self.model.classes_

    @with_masked_split
    def forward(self, X):
        proba = self.model.predict_proba(np.asarray(X, dtype=np.float32))
        self.extras['y_score'] = proba
        return self.classes_[proba.argmax(axis=1)]

    def get_fixed_params(self, inputs):
        return {}

    def get_model_params(self, trial, inputs):
        return {}
