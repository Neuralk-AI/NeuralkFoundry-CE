from dataclasses import dataclass
import numpy as np
from ...workflow import Step
from ...utils.splitting import Split

from ...utils.data import array_to_dataframe


class TabPfnVectorizer(Step):
    """
    Vectorizer that uses TabPFN embeddings for tabular data.

    This step replaces raw input features with embeddings extracted using
    the TabPFN model (either classifier or regressor). Embeddings are computed
    in a transductive way using both training and test folds.

    Attributes
    ----------
    name : str
        Name of the step, set to "tabpfn_vectorizer".

    Inputs
    ------
    X : ndarray
        Input features of shape (n_samples, n_features).
    y : ndarray
        Target labels or values.
    folds : ndarray
        Fold assignment for each sample, must contain values from `Split` enum.
    task : str
        Task type, must be either "classification" or "regression".
    seed : int
        Random seed for reproducibility.

    Outputs
    -------
    X : ndarray
        Transformed feature matrix (embeddings).
    num_features : int
        Dimensionality of the output embeddings.
    """
    name = "tabpfn_vectorizer"

    def __init__(self):
        super().__init__(inputs=['X', 'y', 'folds', 'task', 'seed'], outputs=['X'])

    def _execute(self, inputs: dict):
        """
        Run TabPFN embedding generation on the input data.

        Parameters
        ----------
        inputs : dict
            Dictionary containing:
            - 'X': Input feature matrix
            - 'y': Target values
            - 'folds': Fold assignments for each sample
            - 'task': Task type ('classification' or 'regression')
            - 'seed': Random seed for reproducibility

        Raises
        ------
        ValueError
            If the provided task is not supported (must be 'classification' or 'regression').
        """
        from tabpfn_extensions.embedding import TabPFNEmbedding
        from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor

        X = inputs['X']
        y = inputs['y']
        task = inputs['task']
        seed = inputs['seed']
        folds = inputs['folds']
        if task == 'regression':
            reg = TabPFNRegressor(n_estimators=1, random_seed=seed)
            embedding_extractor = TabPFNEmbedding(tabpfn_reg=reg, n_fold=0)
        elif task == 'classification':
            clf = TabPFNClassifier(n_estimators=1, random_seed=seed)
            embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
        else:
            raise ValueError('Task not suported by the TabPFN vectorizer')

        train_mask = np.isin(np.array(folds), [Split.TRAIN, Split.VAL])
        test_mask = np.isin(np.array(folds), [Split.TEST])

        X_train, y_train, X_test = X[train_mask], y[train_mask], X[test_mask]

        train_embeddings = embedding_extractor.get_embeddings(
            X_train, y_train, X_test,
            data_source="train",
        )

        test_embeddings = embedding_extractor.get_embeddings(
            X_train, y_train, X_test,
            data_source="test",
        )

        embeddings = np.empty((X.shape[0], train_embeddings.shape[1]))
        embeddings[train_mask] = train_embeddings
        embeddings[test_mask] = test_embeddings

        self.output("X", array_to_dataframe(embeddings, prefix="pfn_embedding"))
        self.output("num_features", X.shape[1])