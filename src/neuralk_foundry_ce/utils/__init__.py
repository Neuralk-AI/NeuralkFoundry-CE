"""Utility functions for various uses."""

from .docs import add_submodules_to_docstring
add_submodules_to_docstring(__name__)


from .logging import log

__all__ = [
    'log'
]