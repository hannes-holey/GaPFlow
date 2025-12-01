# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import GaPFlow

project = 'GaPFlow'
copyright = '2025, Hannes Holey'
author = 'Hannes Holey'
version = GaPFlow.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'myst_nb']


pygments_style = 'default'
source_suffix = {'.rst': 'restructuredtext'}

todo_include_todos = False

exclude_patterns = ['conf.py']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'

# -- Extension configuration -------------------------------------------------

# Raise Error (not default Warning) when a notebook execution fails
# (due to code error, timeout, etc.)
# nb_execution_mode = "off"
nb_execution_raise_on_error = False
nb_execution_show_tb = True
nb_execution_timeout = 300
nb_merge_streams = True

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
    'private-members': False,
}

# The following makes mystnb convert notebooks with jupytext
# before execution and documentation rendering. This allows
# storing notebooks in properly versionable text formats, e.g.
# the percent format,
#   https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format
# instead of .ipynb. Also see
#   https://myst-nb.readthedocs.io/en/latest/authoring/custom-formats.html#using-jupytext
nb_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}]
}
