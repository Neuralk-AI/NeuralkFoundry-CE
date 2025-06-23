import sys
import pkgutil


def add_submodules_to_docstring(name, submodules=None):
    """
    Append an autosummary block to the module's __doc__.

    Parameters
    ----------
    name : str
        The full module name (e.g. 'neuralk_foundry_ce.models').
    submodules : list of str, optional
        Explicit list of submodules to include. If None, discovered via pkgutil.
    """
    module = sys.modules.get(name)
    if not module or not hasattr(module, "__path__"):
        return

    if submodules is None:
        submodules = sorted(name for _, name, _ in pkgutil.iter_modules(module.__path__))

    if not submodules:
        return

    module.__doc__ = (module.__doc__ or "") + f"""

.. currentmodule:: {name}

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    
""" + "".join(f"    {sub}\n" for sub in submodules)