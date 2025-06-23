from .base import RegressorModel
from ...utils.splitting import with_masked_split


class TabPFNRegressor(RegressorModel):
    """
    Train a TabPFN regressor on tabular data.

    Inputs
    ------
    - X : Feature matrix for prediction (TabPFN is zero-shot).
    - y : Target labels (used during training interface, but internally not fitted).
    - splits : Optional train/val/test split masks (ignored by TabPFN).

    Outputs
    -------
    - y_pred : Predicted values.

    Parameters
    ----------
    TabPFN does not require traditional hyperparameters. Internally, it uses a pretrained transformer
    model and performs inference directly without fitting.

    Notes
    -----
    Requires `tabpfn` to be installed.
    Only supports tasks with up to 10000 training examples and 500 features.
    """
    name = 'tabpfn-regressor'

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
        from tabpfn import TabPFNRegressor as _TabPFNRegressor

        self.model = _TabPFNRegressor(ignore_pretraining_limits=True, **config)
    
    def get_model_params(self, trial, inputs):
        return {}