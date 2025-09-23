from .base import ClassifierModel
from ...utils.splitting import with_masked_split

class RandForestClassifier(ClassifierModel):
    """
    Apply a RandomForest classifier to tabular data.

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
    name = "randomforest-classifier"

    def __init__(self):
        super().__init__()

    def init_model(self, config):
        from sklearn.ensemble import RandomForestClassifier

        self.config = config
        self.model = RandomForestClassifier(**config)

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return self.model.predict(X)

    def get_fixed_params(self, tags):
        return { 
            "random_state": 42
        }
    
    def get_model_params(self, trial, tags):
        return { 
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
        }
