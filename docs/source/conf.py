# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../../src'))

_here = Path(__file__).parent
shutil.copy(_here.parent.parent / 'README.md', _here / 'README.md')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'arcpy_arrow'
copyright = '2026, Slater Dixon'
author = 'Slater Dixon'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}
autodoc_mock_imports = ['arcpy', 'pyarrow', 'pandas', 'geopandas', 'shapely', 'polars']

templates_path = ['_templates']
exclude_patterns = ['README.md']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
