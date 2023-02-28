# -*- coding: utf-8 -*-
"""Configuration file for the Sphinx documentation builder."""

# tsml documentation master file, created by
# sphinx-quickstart on Wed Dec 14 00:20:27 2022.

import inspect
import os
import sys

import tsml

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

version = tsml.__version__
release = tsml.__version__

github_tag = f"v{version}"

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
on_readthedocs = os.environ.get("READTHEDOCS") == "True"
if not on_readthedocs:
    sys.path.insert(0, os.path.abspath(".."))
else:
    rtd_version = os.environ.get("READTHEDOCS_VERSION")
    if rtd_version == "latest":
        github_tag = "main"


# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tsml"
copyright = "The tsml developers (BSD-3 License)"
author = "Matthew Middlehurst"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "numpydoc",
    "nbsphinx",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", ".ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# auto doc/summary

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

# numpydoc

# see http://stackoverflow.com/q/12206334/562769
numpydoc_show_class_members = True
# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

numpydoc_validation_checks = {"all"}

# nbsphinx

nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_timeout = 600  # seconds, set to -1 to disable timeout

current_file = "{{ env.doc2path(env.docname, base=None) }}"

# add link to original notebook at the bottom and add Binder launch button
# points to latest stable release, not main
notebook_url = f"https://github.com/time-series-machine-learning/tsml-py/tree/{github_tag}/{current_file}"  # noqa
binder_url = f"https://mybinder.org/v2/gh/time-series-machine-learning/tsml-py/{github_tag}?filepath={current_file}"  # noqa
nbsphinx_epilog = f"""
----

Generated using nbsphinx_. The Jupyter notebook can be found here_.
|Binder|_

.. _nbsphinx: https://nbsphinx.readthedocs.io/
.. _here: {notebook_url}
.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: {binder_url}
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


def linkcode_resolve(domain, info):
    """Return URL to source code for sphinx.ext.linkcode."""

    def find_source():
        # try to find the file and line number, used in sktime and tslearn conf.py.
        # originally based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L393
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(tsml.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "tsml/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"

    return "https://github.com/time-series-machine-learning/tsml-py/blob/%s/%s" % (
        version,
        filename,
    )
