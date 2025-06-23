"""Strategies for dividing complex problems into manageable blocks."""

from ...utils.docs import add_submodules_to_docstring
add_submodules_to_docstring(__name__)


from .all import AllPairsBlocking
from .nn import NearestNeighborsBlocking


__all__ = [
    'AllPairsBlocking',
    'NearestNeighborsBlocking'
]