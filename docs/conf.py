import os
import sphinx_rtd_theme
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MIIT'
copyright = '2024, Maximilian Wess'
author = 'Maximilian Wess'
release = '11/02/2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# sys.path.insert(0, os.path.abspath('../miit'))
sys.path.insert(0, os.path.abspath('../../miit'))

extensions = [
    "hoverxref.extension",
    "notfound.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon"
]
# extensions = ["myst_parser"]

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

master_doc = 'index'



# Options for sphinx-hoverxref options
# ------------------------------------

hoverxref_auto_ref = True
hoverxref_role_types = {
    "class": "tooltip",
    "command": "tooltip",
    "confval": "tooltip",
    "hoverxref": "tooltip",
    "mod": "tooltip",
    "ref": "tooltip",
    "reqmeta": "tooltip",
    "setting": "tooltip",
    "signal": "tooltip",
}
hoverxref_roles = ["command", "reqmeta", "setting", "signal"]

html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_style = os.path.join("css", "custom.css")
# # html_favicon = os.path.join("_static", "favicon.ico")
html_static_path = ["_static"]
# html_js_files = ["language_data.js"]

# html_sidebars = {
#     '**': [
#         'installation.html',
#         'usage/config.html',
#         'usage/pairwise.html',
#         'usage/groupwise.html'
#     ]
# }

# html_sidebars = {
#    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
#    'using/windows': ['windowssidebar.html', 'searchbox.html'],
# }

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

exclude_patterns = ['build']
exclude_trees = ['.build']
pygments_style = "sphinx"
