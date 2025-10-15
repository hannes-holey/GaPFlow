# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import hans

project = 'hans'
copyright = '2025, Hannes Holey'
author = 'Hannes Holey'
version = hans.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'myst_nb']


pygments_style = 'sphinx'
source_suffix = {'.rst': 'restructuredtext'}

todo_include_todos = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------

# Raise Error (not default Warning) when a notebook execution fails
# (due to code error, timeout, etc.)
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_execution_timeout = 90

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    ]

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
