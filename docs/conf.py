# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import textwrap

project = 'cutrace'
copyright = '2023, jay-tux'
author = 'jay-tux'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'breathe',
    'exhale',
    'sphinx.ext.graphviz',
    'sphinx.ext.mathjax'
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
    'exhaleDoxygenStdin': textwrap.dedent('''
        INPUT           = ../inc
        PREDEFINED      += __device__=/**@device*/ __host__=/**@host*/
        ALIASES         += host="<para><em>Host function</em><para>"
        ALIASES         += device="<para><em>Device function</em><para>"
        ALIASES         += global="<para><em>Global function</em><para>"
        EXTRACT_ALL     = NO
        EXTRACT_PRIVATE = NO
        EXCLUDE_SYMBOLS = cutrace::impl cutrace::gpu::impl cutrace::cpu::impl cutrace::cpu::schema::impl
    ''')
}

primary_domain = 'cpp'
highlight_language = 'cpp'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'groundwork'
html_static_path = ['_static']
