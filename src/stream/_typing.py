"""Type hints.

This project extensively uses :mod:`~typing` hints.
Note that this is not (necessarily) static typing.
"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, TypeVar, Union

# THIRD PARTY
from numpy import floating
from numpy.typing import NBitBase, NDArray

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.coordinates import BaseCoordinateFrame
    from typing_extensions import TypeAlias


__all__: list[str] = []


##############################################################################
# TYPES
##############################################################################

# -------------------------------------
# NumPy types

N1 = TypeVar("N1", bound=NBitBase)
N2 = TypeVar("N2", bound=NBitBase)

NDFloat = NDArray[floating[N1]]


# -------------------------------------
# Astropy types

FrameLikeType: TypeAlias = Union["BaseCoordinateFrame", str]
