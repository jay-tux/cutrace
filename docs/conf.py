# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cutrace'
copyright = '2023, jay-tux'
author = 'jay-tux'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'breathe',
    'exhale'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

breathe_projects = {
    'cutrace': './_doxygen/xml'
}
breathe_default_project = 'cutrace'

exhale_args = {
    'containmentFolder': './api',
    'rootFileName': 'cutrace_root.rst',
    'doxygenStripFromPath': '..',
    'rootFileTitle': 'cutrace API',
    'createTreeView': True,
    'exhaleExecutesDoxygen': True,
    'exhaleDoxygenStdin': 'INPUT = ../inc'
}

primary_domain = 'cpp'
highlight_language = 'cpp'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
