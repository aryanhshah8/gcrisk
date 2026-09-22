# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add project root so autodoc can import the gcrisk package
sys.path.insert(0, os.path.abspath('..'))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = 'GCR Dosimetry Pipeline'
copyright = '2026, Aryan'
author = 'Aryan'
release = '0.2.0'

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Pull docstrings automatically
    'sphinx.ext.napoleon',      # Support NumPy/Google docstring style
    'sphinx.ext.mathjax',       # Render inline LaTeX math
    'sphinx.ext.viewcode',      # Add "view source" links
    'sphinx.ext.intersphinx',   # Cross-reference external packages
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy', None),
}

# Autodoc options: include both public and private members' docstrings
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'GCR Dosimetry Pipeline v0.2.0'

# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
