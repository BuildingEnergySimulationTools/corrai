import os
import sys
import importlib.metadata

sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "corrai"
copyright = "Nobatek"
author = "Baptiste Durand-Estebe"

# The full version, including alpha/beta/rc tags
version = release = importlib.metadata.version("corrai")
if version is None:
    version = release = "0.0.0"  # only for local dev

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_rtd_theme"

# Theme options
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    # GitHub link
    "github_url": "https://github.com/BuildingEnergySimulationTools/corrai",
    "github_repo": "corrai",
    "github_user": "bdurandestebe",
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "custom.css",
]

# Logo
html_logo = "../logo_corrai.svg"
html_favicon = "../logo_corrai.svg"

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
