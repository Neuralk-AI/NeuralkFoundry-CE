from .base import RegressorModel
from ...utils.splitting import with_masked_split


class CatBoostRegressor(RegressorModel):
    """
    Train a CatBoost regressor on tabular data.

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
    Requires `catboost` to be installed.
    """
    name = 'catboost-regressor'

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
        from catboost import CatBoostRegressor  as _CatBoostRegressor

        self.model = _CatBoostRegressor(verbose=0, **config)

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        return self.model.predict(X)

    def get_model_params(self, trial, inputs):
            return {
                "iterations": trial.suggest_categorical("iterations", [1000]),
                "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [50]),
                "depth": trial.suggest_int("depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_strength": trial.suggest_float("random_strength", 1e-9, 10),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "random_seed": trial._trial_id,
                "loss_function": "RMSE",
                "eval_metric": "RMSE"
            }
