# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CV Robustness Plugin for AIVerify'
copyright = '2026, Gabriel Boey'
author = 'Gabriel Boey'

# -- Path setup --------------------------------------------------------------
# Each algorithm lives in algorithms/<name>_algorithm/<name>_algorithm and is a
# package that uses relative imports (e.g. ``from .cvrob_util import *``). Put
# each algorithm's outer directory on sys.path so autodoc can import the inner
# package as ``<name>_algorithm.<script>``.
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_ALGO_ROOT = os.path.join(_HERE, 'algorithms')
for _name in sorted(os.listdir(_ALGO_ROOT)):
    _pkg_parent = os.path.join(_ALGO_ROOT, _name)
    if os.path.isdir(_pkg_parent):
        sys.path.insert(0, _pkg_parent)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'algorithms', 'node_modules']

# The modules import many heavy / platform-specific third-party libraries
# (including the Unix-only ``resource`` module). Mock them so autodoc can import
# the source for docstrings without the dependencies being installed.
autodoc_mock_imports = [
    'aiverify_test_engine',
    'albumentations',
    'matplotlib',
    'numpy',
    'nrtk',
    'pandas',
    'plotly',
    'pycocotools',
    'PIL',
    'requests',
    'resource',
    'scipy',
    'sklearn',
    'torch',
    'torchmetrics',
    'torchvision',
    'tqdm',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Render Napoleon "Attributes:" sections as inline :ivar: fields instead of
# separate attribute directives. Without this, dataclass fields get documented
# twice (once by Napoleon, once by autodoc's undoc-members), producing
# "duplicate object description" warnings.
napoleon_use_ivar = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary'
]
