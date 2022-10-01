"""Fit a rotated reference frame to stream data."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Generic, TypeVar, final

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from scipy.optimize import OptimizeResult

__all__: list[str] = []


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

    frame: coords.SkyOffsetFrame
    result: R

    @singledispatchmethod
    @classmethod
    def from_result(
        cls: type[FrameOptimizeResult[Any]], optimize_result: object, from_frame: coords.BaseCoordinateFrame | None
    ) -> FrameOptimizeResult[R]:
        if not isinstance(optimize_result, cls):
            raise NotImplementedError(f"optimize_result type {type(optimize_result)} is not known.")

        # overload + Self is implemented here until it works
        if from_frame is not None or from_frame != optimize_result.frame:
            raise ValueError("from_frame must be None or the same as optimize_result's frame")
        return cls(optimize_result.frame, optimize_result.result)

    @from_result.register(OptimizeResult)
    @classmethod
    def _from_result_scipyoptresult(
        cls: type[FrameOptimizeResult[Any]], optimize_result: OptimizeResult, from_frame: coords.BaseCoordinateFrame
    ) -> FrameOptimizeResult[OptimizeResult]:
        # Get coordinates
        optimize_result.x <<= u.deg
        fit_rot, fit_lon, fit_lat = optimize_result.x
        # create SkyCoord
        r = coords.UnitSphericalRepresentation(lon=fit_lon, lat=fit_lat)
        origin = coords.SkyCoord(
            from_frame.realize_frame(r, representation_type=from_frame.representation_type), copy=False
        )
        # transform to offset frame
        fit_frame = origin.skyoffset_frame(rotation=fit_rot)
        fit_frame.representation_type = from_frame.representation_type
        return cls(fit_frame, optimize_result)

    # ===============================================================

    @property
    def rotation(self) -> u.Quantity:
        """The rotation of point on sky."""
        return self.frame.rotation

    @property
    def origin(self) -> coords.BaseCoordinateFrame:
        """The location of point on sky."""
        return self.frame.origin

    # ===============================================================

    def calculate_residual(self, data: coords.SkyCoord, scalar: bool = False) -> u.Quantity:
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
        ur = data.transform_to(self.frame).represent_as(coords.UnitSphericalRepresentation)
        res: u.Quantity = np.abs(ur.lat - 0.0 * u.rad)
        return np.sum(res) if scalar else res
