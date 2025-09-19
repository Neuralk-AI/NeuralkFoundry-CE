from .base import ClassifierModel
from ...utils.splitting import with_masked_split


class TabPFNClassifier(ClassifierModel):
    """
    Train a TabPFN classifier on tabular data.

    Inputs
    ------
    - X : Feature matrix for prediction (TabPFN is zero-shot).
    - y : Target labels (used during training interface, but internally not fitted).
    - splits : Optional train/val/test split masks (ignored by TabPFN).

    Outputs
    -------
    - y_pred : Predicted class labels.
    - y_score : Class probabilities (if available and requested).

    Parameters
    ----------
    TabPFN does not require traditional hyperparameters. Internally, it uses a pretrained transformer
    model and performs inference directly without fitting.

    Notes
    -----
    Only supports classification tasks with up to 10000 training examples, 10 classes, and 500 features.
    For now, detects automatically the device, so we do not load it from config.
    """
    name = 'tabpfn-classifier'

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
        from tabpfn import TabPFNClassifier as _TabPFNClassifier

        self.model = _TabPFNClassifier(ignore_pretraining_limits=True, **config)
    
    def get_fixed_params(self, inputs):
        return {}

    def get_model_params(self, trial, inputs):
        return {}