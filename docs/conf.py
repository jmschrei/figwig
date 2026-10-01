# conf.py
# Sphinx configuration. See https://www.sphinx-doc.org/en/master/usage/configuration.html

import figwig

project = 'figwig'
copyright = '2026, Jacob Schreiber'
author = 'Jacob Schreiber'
release = figwig.__version__
version = release

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',
	'sphinx.ext.intersphinx',
	'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

html_theme = 'sphinx_rtd_theme'
html_static_path = []

autodoc_member_order = 'bysource'
napoleon_numpy_docstring = True
napoleon_google_docstring = False

nbsphinx_execute = 'never'

intersphinx_mapping = {
	'python': ('https://docs.python.org/3', None),
	'numpy': ('https://numpy.org/doc/stable/', None),
}
