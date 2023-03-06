"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

from __future__ import annotations

# STDLIB
import os
from typing import Any

# THIRD PARTY
import pytest

try:
    # THIRD PARTY
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


def pytest_configure(config: pytest.Config) -> None:
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : pytest configuration
    """
    if ASTROPY_HEADER:
        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the
        # tests.
        PYTEST_HEADER_MODULES.pop("Pandas", None)
        PYTEST_HEADER_MODULES["scikit-image"] = "skimage"

        # STDLIB
        # from stream import __version__  # type: ignore
        from importlib.metadata import version

        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = version(packagename)


# ------------------------------------------------------
# doctest fixtures


@pytest.fixture(autouse=True)  # type: ignore
def add_numpy(doctest_namespace: dict[str, Any]) -> None:
    """Add NumPy to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import numpy

    # add to namespace
    doctest_namespace["np"] = numpy


@pytest.fixture(autouse=True)  # type: ignore
def add_astropy(doctest_namespace: dict[str, Any]) -> None:
    """Add Astropy stuff to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import astropy.coordinates as coord
    import astropy.units

    # add to namespace
    doctest_namespace["coord"] = coord
    doctest_namespace["u"] = astropy.units

    # extras
    # THIRD PARTY
    from astropy.visualization import quantity_support, time_support

    quantity_support()
    time_support()
