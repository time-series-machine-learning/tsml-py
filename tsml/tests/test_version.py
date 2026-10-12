"""Tests for the package version."""

import tomllib
from importlib.metadata import version
from pathlib import Path

import tsml


def test_version_is_consistent():
    """Test __version__ matches the version in pyproject.toml and the README."""
    root = Path(tsml.__file__).parent.parent

    # an installed wheel has neither file, only the metadata built from pyproject.toml
    if not (root / "pyproject.toml").is_file():
        assert tsml.__version__ == version("tsml")
        return

    with open(root / "pyproject.toml", "rb") as f:
        assert tsml.__version__ == tomllib.load(f)["project"]["version"]

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"The current release of `tsml` is v{tsml.__version__}." in readme
