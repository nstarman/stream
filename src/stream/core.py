"""Stream arm classes.

Stream arms are descriptors on a `trackstrea.Stream` class.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, SkyCoord  # noqa: TC002
from astropy.io.registry import UnifiedReadWriteMethod
from astropy.units import Quantity
from numpy.lib.recfunctions import structured_to_unstructured
from typing_extensions import TypedDict

# LOCAL
from stream.base import StreamBase
from stream.clean import OUTLIER_DETECTOR_CLASSES, OutlierDetectorBase
from stream.io import (
    StreamArmFromFormat,
    StreamArmRead,
    StreamArmToFormat,
    StreamArmWrite,
)
from stream.utils.coord_utils import get_frame, parse_framelike

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.table import QTable
    from numpy.typing import NDArray

    # LOCAL
    from stream.frame.result import FrameOptimizeResult


__all__ = ["StreamArm"]


##############################################################################
# PARAMETERS


class SupportsFrame(Protocol):
    frame: BaseCoordinateFrame


Self = TypeVar("Self", bound=SupportsFrame)  # from typing_extensions import Self


class _StreamArmCache(TypedDict):
    """Cache for Stream."""

    # frame
    frame_fit_result: FrameOptimizeResult[Any] | None


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class StreamArm(StreamBase):
    """An arm of a stream"""

    _CACHE_CLS: ClassVar[type] = _StreamArmCache

    data: QTable
    origin: SkyCoord
    frame: BaseCoordinateFrame | None = None
    name: str | None = None
    prior_cache: InitVar[dict[str, Any] | None] = None

    def __post_init__(self, prior_cache: dict[str, Any] | None) -> None:
        # frame-like -> frame
        object.__setattr__(self, "frame", None if self.frame is None else parse_framelike(self.frame))

        # Cache
        self._cache: dict[str, Any]
        object.__setattr__(self, "cache", prior_cache)

    read = UnifiedReadWriteMethod(StreamArmRead)
    write = UnifiedReadWriteMethod(StreamArmWrite)
    to_format = UnifiedReadWriteMethod(StreamArmToFormat)
    from_format = UnifiedReadWriteMethod(StreamArmFromFormat)

    # ===============================================================
    # Directly from Data

    def get_mask(self, minPmemb: Quantity | None = None, include_order: bool = True) -> NDArray[np.bool_]:
        """Which elements of the stream are masked."""
        minPmemb = self.flags.minPmemb if minPmemb is None else minPmemb
        mask = (
            (self.data["Pmemb"] < minPmemb)
            & self.data["Pmemb"].mask
            & (self.data["order"].mask if include_order else True)
        )
        return cast("np.ndarray[Any, np.dtype[np.bool_]]", mask)

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Full mask: minPmemb & include_order"""
        return self.get_mask(minPmemb=None, include_order=True)

    @property
    def data_coords(self) -> SkyCoord:
        """Get ``coord`` from data table."""
        return self.data["coord"][~self.mask]

    @property
    def data_frame(self) -> BaseCoordinateFrame:
        """The `astropy.coordinates.BaseCoordinateFrame` of the data."""
        reptype = self.data["coord"].representation_type
        if not self.has_distances:
            reptype = getattr(reptype, "_unit_representation", reptype)

        # Get the frame from the data
        frame: BaseCoordinateFrame = self.data["coord"].frame.replicate_without_data(representation_type=reptype)

        return frame

    # ===============================================================
    # System stuff (fit dependent)

    @property
    def _best_frame(self) -> BaseCoordinateFrame:
        """:attr:`Stream.frame` unless its `None`, else :attr:`Stream.data_frame`."""
        frame = self.frame if self.frame is not None else self.data_frame
        return frame

    @property
    def coords(self) -> SkyCoord:
        """Data coordinates transformed to `Stream.frame` (if there is one)."""
        order: np.ndarray[Any, np.dtype[np.bool_]] = self.data["order"][~self.data["order"].mask]

        dc = cast("SkyCoord", self.data_coords[order])
        frame = self._best_frame

        c = dc.transform_to(frame)
        c.representation_type = frame.representation_type
        c.differential_type = frame.differential_type
        return c

    # ===============================================================
    # Cleaning Data

    # TODO! change to added property
    def mask_outliers(self, outlier_method: str | OutlierDetectorBase = "scipyKDTreeLOF", **kwargs: Any) -> None:
        """Detect and label outliers, masking their Pmemb and order."""
        # TODO! more complete, including velocity data
        data = structured_to_unstructured(self.data_coords.data._values)

        if isinstance(outlier_method, str):
            outlier_method = OUTLIER_DETECTOR_CLASSES[outlier_method]()
        elif not isinstance(outlier_method, OutlierDetectorBase):
            raise TypeError

        isoutlier = outlier_method.fit_predict(data, data, **kwargs)

        self.data["order"].mask = isoutlier
        # TODO? also mask Pmemb?

    # ===============================================================
    # Convenience Methods

    def fit_frame(
        self: StreamArm,
        rot0: Quantity[u.deg] | None = Quantity(0, u.deg),
        *,
        force: bool = False,
        **kwargs: Any,
    ) -> StreamArm:
        """Convenience method to fit a frame to the data.

        The frame is an on-sky rotated frame.
        To prevent a frame from being fit, the desired frame should be passed
        to the Stream constructor at initialization.

        Parameters
        ----------
        rot0 : |Quantity| or None.
            Initial guess for rotation.

        force : bool, optional keyword-only
            Whether to force a frame fit. Default `False`.
            Will only fit if a frame was not specified at initialization.

        **kwargs : Any
            Passed to fitter.

        Returns
        -------
        StreamArm
            A copy, with `frame` replaced.

        Raises
        ------
        ValueError
            If a frame has already been fit and `force` is not `True`.
        TypeError
            If a system frame was given at initialization.
        """
        if self.frame is not None and not force:
            raise TypeError("a system frame was given at initialization. Use ``force`` to re-fit.")

        # LOCAL
        from stream.frame.fit import fit_frame

        newstream: StreamArm = fit_frame(self, rot0=rot0, **kwargs)
        return newstream

    # ===============================================================
    # Misc

    def __base_repr__(self, max_lines: int | None = None) -> list[str]:
        rs = super().__base_repr__(max_lines=max_lines)

        # 5) data table
        datarep: str = self.data._base_repr_(html=False, max_width=None, max_lines=max_lines)
        table: str = "\n\t".join(datarep.split("\n")[1:])
        rs.append("  Data:\n\t" + table)

        return rs


@get_frame.register
def _get_frame_streamarm(stream: StreamArm, /) -> BaseCoordinateFrame:
    if stream.frame is None:
        raise ValueError("need to fit a frame")
    return stream.frame
