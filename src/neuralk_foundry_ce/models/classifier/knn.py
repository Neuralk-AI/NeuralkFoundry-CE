from .base import ClassifierModel
from ...utils.splitting import with_masked_split

class KNNClassifier(ClassifierModel):
    """
    Apply a KNN classifier to tabular data.

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
    Requires `scikit-learn` to be installed.
    """  
    name = "knn-classifier"

    def __init__(self):
        super().__init__()
        self.n_ensemble = 50

    def init_model(self, config):
        from sklearn.neighbors import KNeighborsClassifier

        self.config = config
        self.model = KNeighborsClassifier(**config)

    @with_masked_split
    def train(self, X, y):
        self.model.fit(X, y)

    @with_masked_split
    def forward(self, X):
        self.extras['y_score'] = self.model.predict_proba(X)
        return self.model.predict(X)

    def get_fixed_params(self, tags):
        return { }
    
    def get_model_params(self, trial, tags):
        return { 
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15, step=2),
            "p": trial.suggest_int("p", 1, 2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
