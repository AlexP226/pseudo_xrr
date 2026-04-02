#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import sphinx_rtd_theme

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# If import fails during docs build, keep docs config importable
try:
    import pseudo_xrr
    VERSION = pseudo_xrr.__version__
except Exception:
    VERSION = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_copybutton",
]

plot_html_show_source_link = False
plot_html_show_formats = False
autosummary_generate = True
numpydoc_show_class_members = False

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "Pseudo XRR"
copyright = "2025, Brookhaven National Lab"
author = "Brookhaven National Lab"

version = VERSION
release = VERSION

language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = False

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "relations.html",
        "searchbox.html",
    ]
}

htmlhelp_basename = "pseudo_xrr"

latex_elements = {}

latex_documents = [
    (
        master_doc,
        "pseudo_xrr.tex",
        "Pseudo XRR Documentation",
        "Contributors",
        "manual",
    ),
]

man_pages = [
    (
        master_doc,
        "pseudo_xrr",
        "Pseudo XRR Documentation",
        [author],
        1,
    )
]

texinfo_documents = [
    (
        master_doc,
        "pseudo_xrr",
        "Pseudo XRR Documentation",
        author,
        "pseudo_xrr",
        "Python package for pseudo X-ray reflectivity",
        "Miscellaneous",
    ),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}