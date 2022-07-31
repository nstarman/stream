"""Fit a Rotated reference frame."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import (
    BaseCoordinateFrame,
    SkyCoord,
    SkyOffsetFrame,
    UnitSphericalRepresentation,
)
from scipy.optimize import OptimizeResult

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.units import Quantity

__all__ = ["FrameOptimizeResult"]


##############################################################################
# Parameters

R = TypeVar("R")


##############################################################################
# CODE
##############################################################################


@final
@dataclass(frozen=True, repr=True)
class FrameOptimizeResult(Generic[R]):
    """Result of fitting a rotated frame.

    Parameters
    ----------
    origin : `~astropy.coordinates.SkyCoord`
        The location of point on sky about which to rotate.
    rotation : Quantity['angle']
        The rotation about the ``origin``.
    **kwargs : Any
        Fit results. See `~scipy.optimize.OptimizeResult`.
    """

    frame: SkyOffsetFrame
    result: R

    @singledispatchmethod
    @classmethod
    def from_result(
        cls: type[FrameOptimizeResult[Any]], optimize_result: object, from_frame: BaseCoordinateFrame
    ) -> FrameOptimizeResult[R]:
        if not isinstance(optimize_result, cls):
            raise NotImplementedError(f"optimize_result type {type(optimize_result)} is not known.")

        # overload + Self is implemented here until it works
        if from_frame is not None and optimize_result.frame != from_frame:
            raise ValueError
        return cls(optimize_result.frame, optimize_result.result)

    @from_result.register(OptimizeResult)
    @classmethod
    def _from_result_scipyoptresult(
        cls: type[FrameOptimizeResult[Any]], optimize_result: OptimizeResult, from_frame: BaseCoordinateFrame
    ) -> FrameOptimizeResult[OptimizeResult]:
        # Get coordinates
        optimize_result.x <<= u.deg
        fit_rot, fit_lon, fit_lat = optimize_result.x
        # create SkyCoord
        r = UnitSphericalRepresentation(lon=fit_lon, lat=fit_lat)
        origin = SkyCoord(from_frame.realize_frame(r, representation_type=from_frame.representation_type), copy=False)
        # transform to offset frame
        fit_frame = origin.skyoffset_frame(rotation=fit_rot)
        fit_frame.representation_type = from_frame.representation_type
        return cls(fit_frame, optimize_result)

    # ===============================================================

    @property
    def rotation(self) -> Quantity:
        """The rotation of point on sky."""
        return self.frame.rotation

    @property
    def origin(self) -> BaseCoordinateFrame:
        """The location of point on sky."""
        return self.frame.origin

    # ===============================================================

    def calculate_residual(self, data: SkyCoord, scalar: bool = False) -> Quantity:
        """Calculate result residual given the fit frame.

        Parameters
        ----------
        data : (N,) `~astropy.coordinates.SkyCoord`
        scalar : bool
            Whether to sum the results.

        Returns
        -------
        `~astropy.units.Quantity`
            Scalar if ``scalar``, else length N.
        """
        ur = data.transform_to(self.frame).represent_as(UnitSphericalRepresentation)
        res: Quantity = np.abs(ur.lat - 0.0 * u.rad)
        return np.sum(res) if scalar else res
