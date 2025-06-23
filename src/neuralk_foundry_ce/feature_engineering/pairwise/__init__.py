"""Feature engineering for sample pairs."""

from ...utils.docs import add_submodules_to_docstring
add_submodules_to_docstring(__name__)


from .pair_feature_generator import PairFeatureGenerator

__all__ = [
    'PairFeatureGenerator',
]