from sklearn.preprocessing import LabelEncoder as _LabelEncoder
from sklearn.preprocessing import TargetEncoder
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ...workflow import Step, Field


class Preprocessing(Step):
    name = "preprocessing"
    # Preprocessing steps are cheap and deterministic; skip their on-disk cache
    # to keep workflow outputs compact (only the dataset + splits + model
    # results are worth persisting).
    cache_outputs = False


class ColumnTypeDetection(Preprocessing):
    """
    Detects column types and forwards them to the rest of the pipeline.

    This step analyzes the input DataFrame and classifies columns by type:
    numerical, categorical, text, and date. These column groupings can then be
    used by downstream steps for appropriate preprocessing.

    Parameters
    ----------
    X : pandas.DataFrame
        The raw input feature table.

    Returns
    -------
    numerical_features : list of str
        Names of columns identified as numerical features.

    categorical_features : list of str
        Names of columns identified as categorical features.

    text_features : list of str
        Names of columns identified as text features.

    date_features : list of str
        Names of columns identified as date or datetime features.
    """
    name = "column-type-detection"
    inputs = [
        Field('X', 'Input features of the dataset'),
        Field('y', 'Target variable to predict'),
    ]
    outputs = [
        Field('numerical_features', 'Names of the numerical feature columns'),
        Field('categorical_features', 'Names of the categorical feature columns'),
        Field('text_features', 'Names of the text feature columns'),
        Field('date_features', 'Names of the date feature columns'),
    ]
    params = [
        Field('numerical_features', 'Names of the numerical feature columns', required=False),
        Field('categorical_features', 'Names of the categorical feature columns', required=False),
        Field('text_features', 'Names of the text feature columns', required=False),
        Field('date_features', 'Names of the date feature columns', required=False),      
    ]

    def _infer_feature_types(self, X: pd.DataFrame):
        """
        Infer feature types from a DataFrame for preprocessing.

        This method categorizes the input columns into four types:
        numerical, categorical, text, and date. The classification rules are:
        
        - Numerical: columns with numeric dtype, excluding datetime types.
        - Categorical: object, string, or category dtype with low cardinality (<=30 unique values).
        - Text: object or string dtype with high cardinality (>30 unique values).
        - Date: datetime or datetimetz columns.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data from which feature types are inferred.

        Returns
        -------
        numerical : list of str
            Names of columns classified as numerical features.

        categorical : list of str
            Names of columns classified as categorical features.

        text : list of str
            Names of columns classified as text features.

        date : list of str
            Names of columns classified as date features.
        """
        # Date features
        date_cols = X.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # Text features: object/string with high cardinality (e.g. >30 unique values)
        text_cols = [
            col for col in X.select_dtypes(include=["object", "string"]).columns
            if col not in date_cols and X[col].nunique(dropna=True) > 30
        ]

        # Categorical features: non-numeric, non-date, non-text, low cardinality
        cat_cols = [
            col for col in X.select_dtypes(include=["category", "object", "string"]).columns
            if col not in text_cols and col not in date_cols
        ]

        # Numerical features: numeric columns excluding dates
        num_cols = [
            col for col in X.select_dtypes(include=["number"]).columns
            if col not in date_cols
        ]

        return num_cols, cat_cols, text_cols, date_cols

    def _execute(self, inputs: dict):
        X = inputs["X"]

        inferred_num, inferred_cat, inferred_text, inferred_date = self._infer_feature_types(X)

        if self.numerical_features:
            num_feats = self.numerical_features
        else:
            num_feats = inferred_num
        self.output('numerical_features', self.numerical_features or inferred_num)
        self.output('categorical_features', self.categorical_features or inferred_cat)
        self.output('text_features', self.text_features or inferred_text)
        self.output('date_features', self.date_features or inferred_date)


class LabelEncoder(Step):
    """
    Encode class labels as integers using scikit-learn's LabelEncoder.

    This step takes a target vector `y` and applies label encoding, transforming
    each unique class label to a corresponding integer.

    Inputs
    ------
    y : array-like
        Target labels to encode (e.g., list, NumPy array, or Pandas Series).

    Outputs
    -------
    y : np.ndarray
        Integer-encoded labels in a NumPy array.
    """
    name = "label-encoding"
    cache_outputs = False
    inputs = [Field('y', 'Target variable to predict')]
    outputs = [
        Field('y', 'Encoded target variable'),
        Field('y_classes', 'Mapping int -> class')
    ]

    def _execute(self, inputs: dict):
        y = inputs["y"]
        encoder = _LabelEncoder()
        y = encoder.fit_transform(y)
        self.output('y', y)
        self.output('y_classes', encoder.classes_)
