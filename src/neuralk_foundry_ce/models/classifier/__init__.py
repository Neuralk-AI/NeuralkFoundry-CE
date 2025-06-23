"""Models dedicated to classification tasks."""

from ...utils.docs import add_submodules_to_docstring
add_submodules_to_docstring(__name__)


from .xgboost import XGBoostClassifier
from .catboost import CatBoostClassifier
from .lightgbm import LightGBMClassifier
from .tabpfn import TabPFNClassifier
from .base import ClassifierModel
from .tabicl import TabICLClassifier
from .mlp import MLPClassifier


__all__ = [
    'XGBoostClassifier',
    'CatBoostClassifier',
    'LightGBMClassifier',
    'TabPFNClassifier',
    'TabICLClassifier',
    'ClassifierModel',
    'MLPClassifier',
]