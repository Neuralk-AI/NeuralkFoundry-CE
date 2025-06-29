# -*- coding: utf-8 -*-


import os
import re
import sys
from datetime import date

import requests
from matplotlib import use

from neuralk_foundry_ce import __version__

year = date.today().strftime("%Y")
use("agg")

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../src/'))
# sys.path.insert(0, os.path.abspath('./'))
# sys.path.insert(0, os.path.abspath("../src/neuralk_foundry_ce"))

# -- Load extensions ------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
    "numpydoc",
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
]

# -- General configuration ------------------------------------------------------------

templates_path = ["_templates"]
source_suffix = ".rst"
source_encoding = "utf-8"

master_doc = "index"  # The master toctree document.

# General information about the project.
project = 'Neuralk Foundry'
copyright = "{}, Neuralk-AI".format(year)
author = "The Neuralk-AI team"

# The version.
version = __version__
release = __version__
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
]

add_function_parentheses = False
add_module_names = False
show_authors = False  # section and module author directives will not be shown
todo_include_todos = False  # Do not show TODOs in docs


# -- Options for HTML output ----------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_short_title = "Foundry"
html_favicon = "_static/favicon.png"
html_css_files = ["css/custom.css"]
html_show_sphinx = True
html_show_copyright = True
htmlhelp_basename = "Neuralkdoc"  # Output file base name for HTML help builder.
html_use_smartypants = True
html_show_sourcelink = True

html_theme_options = {
    "use_edit_page_button": True,
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",  # Label for this link
            "url": "https://github.com/Neuralk-AI/NeuralkFoundry-CE",  # required
            "icon": "fab fa-github-square",
            "type": "fontawesome",  # Default is fontawesome
        }
    ],
    "announcement": "TODO",
}

html_context = {
    "github_user": "NeuralkAI",
    "github_repo": "NeuralkFoundry-CE",
    "github_version": "master",
    "doc_path": "docs",
}

html_sidebars = {
    "map": [],  # Test what page looks like with no sidebar items
}

# -- Napoleon settings ----------------------------------------------------------------

napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_type_aliases = {
    "array-like": ":term:`array-like <array_like>`",
    "array_like": ":term:`array_like`",
    "foundry": "neuralk_foundry_ce",
    "TimestampType": "pandas.Timestamp",
}
# add custom section to docstrings in Parameters style
napoleon_custom_sections = [
    ("Inputs", "params_style"),
    ("Outputs", "returns_style"),
]


# -- Autodoc, autosummary, and autosectionlabel settings ------------------------------



def skip_reexported_symbols(app, what, name, obj, skip, options):
    declared_mod = name.rsplit(".", 1)[0] if "." in name else None
    real_mod = getattr(obj, "__module__", None)

    if not declared_mod or not real_mod:
        return None

    # Skip if re-exported from a different module and NOT internal
    if declared_mod != real_mod and not real_mod.startswith("neuralk_foundry_ce"):
        print(f"🔁 Skipping external re-export: {name} (from {real_mod})")
        return True

    return None

def setup(app):
    app.connect("autodoc-skip-member", skip_reexported_symbols)




autodoc_typehints = "description"
autodoc_typehints_format = "short"

autosummary_generate = True

autoclass_content = "class"

autosectionlabel_prefix_document = True


autodoc_default_options = {
    "members": True,
    "imported-members": True,
}

# -- Numpydoc settings ----------------------------------------------------------------

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

# -- Set intersphinx Directories ------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- myst_nb options ------------------------------------------------------------------

nb_execution_allow_errors = True  # Allow errors in notebooks, to see the error online
nb_execution_mode = "auto"
nb_merge_streams = True
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_dmath_double_inline = True

