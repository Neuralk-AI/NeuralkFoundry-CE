from .base import BaseVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
import pandas as pd
import numpy as np


class TfidfVectorizer(BaseVectorizer):
    """
    Encode string columns in a DataFrame using TF-IDF vectorization.
    
    Attributes
    ----------
    name : str
        Name identifier of the vectorizer. Set to "tfidf-vectorizer".

    Methods
    -------
    forward(X : pd.DataFrame, y=None) -> pd.DataFrame
        Transform the input DataFrame by applying TF-IDF encoding to string columns.
        Non-text columns are retained as-is.
    
    Notes
    -----
    - The number of TF-IDF features per column is limited to `max_features=20`.
    - Missing values in text columns are replaced with empty strings before vectorization.
    - Column names are prefixed with the original column name to avoid conflicts.
    """

    name = "tfidf-vectorizer"

    def __init__(self):
        super().__init__()

    def forward(self, X: pd.DataFrame, y=None):
        text_columns = [
            col for col in X.columns
            if X[col].dtype == 'object' and X[col].apply(lambda x: isinstance(x, str)).any()
        ]

        non_text_columns = [col for col in X.columns if col not in text_columns]
        df_parts = []

        # Transform text columns
        for col in text_columns:
            X[col] = X[col].fillna("")
            vec = _TfidfVectorizer(max_features=20)
            X_col_trans = vec.fit_transform(X[col])
            col_names = [f"{col}__{name}" for name in vec.get_feature_names_out()]
            df_col = pd.DataFrame(X_col_trans.toarray(), columns=col_names, index=X.index)
            df_parts.append(df_col)

        # Append non-text columns as-is
        df_parts.append(X[non_text_columns].reset_index(drop=True))
        X = pd.concat(df_parts, axis=1)
        return X