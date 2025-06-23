"""Model components for classification, regression, embedding, and more."""

from ..utils.docs import add_submodules_to_docstring
add_submodules_to_docstring(__name__)

from .base import BaseModel
from .classifier import ClassifierModel

__all__ = [
    'BaseModel',
    'ClassifierModel',
]
