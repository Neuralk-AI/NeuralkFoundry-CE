"""Feature engineering steps producing embeddings."""

from ...utils.docs import add_submodules_to_docstring
add_submodules_to_docstring(__name__)


from .identity import IdentityVectorizer
from .tablevectorizer import TableVectorizer
from .tabpfnvectorizer import TabPfnVectorizer
from .tfidfencoder import TfidfVectorizer
from .textencoder import TextVectorizer


__all__ = [
    'IdentityVectorizer',
    'TableVectorizer',
    'TabPfnVectorizer',
    'TfidfVectorizer',
    'TextVectorizer',
]