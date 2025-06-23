"""Workflow steps for data grouping and clustering."""

from ...utils.docs import add_submodules_to_docstring
add_submodules_to_docstring(__name__)


from .connected_components import ComponentExtractor

__all__ = [
    'ComponentExtractor',
]