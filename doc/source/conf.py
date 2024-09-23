# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from sphinx.config import is_serializable

import pyrasa

project = 'PyRASA'
copyright = '2024, Fabian Schmidt, Thomas Hartmann'
author = 'Fabian Schmidt, Thomas Hartmann'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    #'sphinx.ext.githubpages',
    #'sphinx.ext.intersphinx',
    #'sphinx.ext.viewcode',
    #'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]

templates_path = ['_templates']
exclude_patterns = ['auto_examples/index.rst', '_build', 'Thumbs.db', '.DS_Store', 'generated']

source_suffix = ['.rst', '.md']

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = pyrasa.__version__
# The full version, including alpha/beta/rc tags.
release = version


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

switcher_version_match = 'dev' if 'dev' in release else version

html_theme_options = {
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
    'navbar_end': ['theme-switcher', 'version-switcher', 'navbar-icon-links'],
    'switcher': {
        'json_url': 'https://raw.githubusercontent.com/mne-tools/mne-bids/main/doc/_static/versions.json',  # noqa: E501
        'version_match': switcher_version_match,
    },
}

sphinx_gallery_conf = {
    'doc_module': 'pyrasa',
    'reference_url': {
        'pyrasa': None,
    },
    'backreferences_dir': 'generated',
    'examples_dirs': '../../examples',
    'within_subsection_order': 'ExampleTitleSortKey',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': '^((?!sgskip).)*$',
}

assert is_serializable(sphinx_gallery_conf)
