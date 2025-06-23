"""Workflow steps for regression tasks."""

from ...utils.docs import add_submodules_to_docstring
add_submodules_to_docstring(__name__)


from .xgboost import XGBoostRegressor
from .catboost import CatBoostRegressor
from .lightgbm import LightGBMRegressor


__all__ = [
    'XGBoostRegressor',
    'CatBoostRegressor',
    'LightGBMRegressor',
]