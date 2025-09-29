from .base import BaseVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
import pandas as pd
import numpy as np
from ...workflow import Field


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
    """

    name = "tfidf-vectorizer"

    inputs = [
        Field('X', 'Input features of the dataset'),
        Field('text_features', 'Names of the text feature columns'),
    ]

    def __init__(self):
        super().__init__()


    def _execute(self, inputs: dict):
        X = inputs['X']

        text_columns = inputs['text_features']

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
    
        self.output("X", X)