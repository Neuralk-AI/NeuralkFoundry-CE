from .base import RegressorModel
from ...utils.splitting import with_masked_split


class LightGBMRegressor(RegressorModel):
    """
    Train a LightGBM regressor on tabular data.

    Inputs
    ------
    - X : Feature matrix for training or prediction.
    - y : Target labels (for training only).
    - splits : Optional train/val/test split masks.

    Outputs
    -------
    - y_pred : Predicted values.

    Notes
    -----
    Requires `lightgbm` to be installed.
    """
    name= "lightgbm-regressor"

    def __init__(self):
        super().__init__()

    def init_model(self, config):
        """
        Initialize the model with given hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments corresponding to model hyperparameters.
            These will be passed directly to the model constructor.
        """
        from lightgbm import LGBMRegressor
        
        self.model = LGBMRegressor(**config)

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        return self.model.predict(X)

    def get_model_params(self, trial, inputs):
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 500, 1000]),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": trial._trial_id,
            "objective": "regression",
            "metric": "rmse"
        }
