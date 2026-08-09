# Configuration file for the Sphinx documentation builder.
#
# Build locally with:
#     python3 -m venv /tmp/docsvenv
#     /tmp/docsvenv/bin/pip install -r docs/requirements.txt
#     /tmp/docsvenv/bin/sphinx-build -W -b html docs docs/_build/html
#
# There is deliberately no autodoc here. Documenting the C++ would need Doxygen
# plus Breathe, and the headers carry ordinary comments rather than Doxygen
# markup, so it would emit bare signatures with no prose. Documenting the Python
# module with autodoc would need the pybind11 extension importable at build time,
# which means SUNDIALS, netCDF and Eigen on the Read the Docs builder. Both
# interfaces are therefore written by hand, and `docs/` is laid out so an api/
# section could be added later without moving anything.

project = "MaNTA"
copyright = "University of Maryland; see the COPYRIGHT file"
author = "Myles Kelly, Ian Abel, Eddie Tocco"

# MaNTA does not carry a version number anywhere in the tree -- there is no
# VERSION file and no version macro -- so there is nothing to read one from.
# Leave it unset rather than invent a number that would immediately be wrong.
release = ""

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

# sphinx_material.get_html_context() returns a dict containing functions, which
# Sphinx cannot pickle into its environment cache, so it warns once per build:
#   cannot cache unpickable configuration value: 'html_context'
# It is harmless -- the context is rebuilt every run anyway -- but it is fatal
# under -W, and the docs are built with -W so that a broken cross-reference fails
# rather than scrolls past. Suppress just that one category.
suppress_warnings = ["config.cache"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "requirements.txt"]

# The default role, so `like this` renders as literal rather than as a broken
# cross-reference. Most inline markup in these pages is a config key, a file
# name or a C++ identifier.
default_role = "literal"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML output -------------------------------------------------------------

import sphinx_material  # noqa: E402

html_theme = "sphinx_material"
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()

html_theme_options = {
    "nav_title": "MaNTA",
    "color_primary": "indigo",
    "color_accent": "blue",
    "repo_url": "https://github.com/ianabel/MaNTA",
    "repo_name": "MaNTA",
    "repo_type": "github",
    "globaltoc_depth": 2,
    "globaltoc_collapse": True,
    "globaltoc_includehidden": True,
    "master_doc": False,
}

# sphinx-material needs these four sidebars; it ships the templates and renders
# an empty navigation column without them.
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

html_static_path = ["_static"]
html_title = "MaNTA"
html_short_title = "MaNTA"

# Where this is published, so generated pages carry a canonical link rather than
# letting the per-version URLs compete with each other in search results.
html_baseurl = "https://manta-docs.readthedocs.io/en/latest/"
