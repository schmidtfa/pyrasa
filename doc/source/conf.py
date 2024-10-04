# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


from sphinx.config import is_serializable

import pyrasa

# import pyrasa.irasa_mne

project = 'PyRASA'
copyright = '2024, Fabian Schmidt, Thomas Hartmann'
author = 'Fabian Schmidt, Thomas Hartmann'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.mathjax',  # optional, if you need to render math
    'sphinx.ext.viewcode',
    'nbsphinx',
]


numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_aliases = {
    'array-like': ':term:`array_like <numpy:array_like>`',
    'int': ':class:`int <python:int>`',
    'bool': ':class:`bool <python:bool>`',
    'float': ':class:`float <python:float>`',
    'list': ':class:`list <python:list>`',
    'tuple': ':class:`tuple <python:tuple>`',
}
numpydoc_xref_ignore = {
    # words
    'instance',
    'instances',
    'of',
}


# generate autosummary even if no references
autosummary_generate = True
autodoc_default_options = {'inherited-members': None}
default_role = 'autolink'  # XXX silently allows bad syntax, someone should fix

exclude_patterns = ['auto_examples/index.rst', '_build', 'Thumbs.db', '.DS_Store', 'generated']

html_show_sourcelink = False
html_copy_source = False

html_theme = 'pydata_sphinx_theme'


templates_path = ['_templates']
html_static_path = ['_static']
# html_css_files = ["style.css"]

source_suffix = ['.rst', '.md']
# master_doc = "index"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = 'dev' if 'dev' in pyrasa.__version__ else pyrasa.__version__
# The full version, including alpha/beta/rc tags.
release = version

html_title = 'PyRASA' + ' ' + release
html_short_title = 'PyRASA'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme_options = {
    #'navbar_sidebarrel': False,
    'icon_links': [
        dict(
            name='GitHub',
            url='https://github.com/schmidtfa/pyrasa',
            icon='fab fa-github-square',
        ),
    ],
    'icon_links_label': 'Quick Links',  # for screen reader
    'use_edit_page_button': False,
    'navigation_with_keys': False,
    'show_toc_level': 1,
    'header_links_before_dropdown': 6,
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
}

html_context = {
    'default_mode': 'auto',
    'doc_path': 'doc',
}

html_sidebars = {}


html_short_title = 'PyRASA'


sphinx_gallery_conf = {
    'doc_module': 'pyrasa',
    'reference_url': {
        'pyrasa': None,
    },
    'backreferences_dir': 'generated',
    'examples_dirs': 'examples',
    'within_subsection_order': 'ExampleTitleSortKey',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': '^((?!sgskip).)*$',
}

assert is_serializable(sphinx_gallery_conf)
