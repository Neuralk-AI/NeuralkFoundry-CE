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
from .modern_nca import ModernNCAClassifier
from .randomforest import RandForestClassifier
from .knn import KNNClassifier
from .realmlp import RealMLPClassifier
from .tabm import TabMClassifier


__all__ = [
    'XGBoostClassifier',
    'CatBoostClassifier',
    'LightGBMClassifier',
    'TabPFNClassifier',
    'TabICLClassifier',
    'ClassifierModel',
    'MLPClassifier',
    'ModernNCAClassifier',
    'RandForestClassifier',
    'KNNClassifier',
    'RealMLPClassifier',
    'TabMClassifier',
]