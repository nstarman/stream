"""Stream arm classes.

Stream arms are descriptors on a `trackstrea.Stream` class.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, TypeVar, cast

# THIRD PARTY
from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.coordinates import concatenate as concatenate_coords
from astropy.table import Column, QTable

# LOCAL
from stream.base import CollectionBase, StreamBase
from stream.core import StreamArm
from stream.utils.coord_utils import get_frame, parse_framelike
from stream.utils.descriptors.cache import CacheProperty

if TYPE_CHECKING:
    # LOCAL
    from stream._typing import FrameLikeType
    from stream.clean.base import OutlierDetectorBase
    from stream.frame.fit.result import FrameOptimizeResult


__all__ = ["StreamArmsBase", "StreamArms", "Stream"]


##############################################################################
# PARAMETERS

Self = TypeVar("Self", bound="CollectionBase")  # type: ignore  # from typing_extensions import Self


class _StreamCache(TypedDict):
    """Cache for Stream."""

    # frame
    frame_fit_result: FrameOptimizeResult | None


##############################################################################


class StreamArmsBase(CollectionBase[StreamArm]):
    """Base class for a collection of stream arms."""


#####################################################################


class StreamArms(StreamArmsBase):
    """A collection of stream arms.

    See Also
    --------
    Stream
        An object that brings together 2 stream arms, but acts like 1 stream.
    """


#####################################################################


class Stream(StreamArmsBase, StreamBase):
    """A Stellar Stream.

    Parameters
    ----------
    data : `~astropy.table.Table`
        The stream data.

    origin : `~astropy.coordinates.ICRS`
        The origin point of the stream (and rotated reference frame).

    data_err : `~astropy.table.QTable` (optional)
        The data_err must have (at least) column names
        ["x_err", "y_err", "z_err"]

    frame : `~astropy.coordinates.BaseCoordinateFrame` or None, optional keyword-only
        The stream frame. Locally linearizes the data.
        If `None` (default), need to fit for the frame.

    name : str or None, optional keyword-only
        The name fo the stream.
    """

    _CACHE_CLS: ClassVar[type] = _StreamCache
    cache = CacheProperty["StreamBase"]()

    def __init__(
        self, data: dict[str, StreamArm] | None = None, /, *, name: str | None = None, **kwargs: StreamArm
    ) -> None:

        super().__init__(data, name=name, **kwargs)

        cache = CacheProperty._init_cache(self)
        self._cache: dict[str, Any]
        object.__setattr__(self, "_cache", cache)

        # validate data length
        if len(self._data) > 2:
            raise NotImplementedError(">2 stream arms are not yet supported")

        # validate that all the data frames are the same
        data_frame = self.data_frame
        origin = self.origin
        frame = self.frame
        for name, arm in self.items():
            if not arm.data_frame == data_frame:
                raise ValueError(f"arm {name} data-frame must match {data_frame}")

            if not arm.origin == origin:
                raise ValueError(f"arm {name} origin must match {origin}")

            if not arm.frame == frame:
                raise ValueError(f"arm {name} origin must match {frame}")

    @classmethod
    def from_data(
        cls: type[Self],
        data: QTable,
        origin: SkyCoord,
        *,
        name: str | None = None,
        data_err: QTable | None = None,
        frame: FrameLikeType | None = None,
        caches: dict[str, dict[str, Any] | None] | None = None,
    ) -> Self:
        # split data by arm
        data = data.group_by("tail")
        data.add_index("tail")

        # similarly for errors
        if data_err is not None:
            data_err = cast(QTable, data_err.group_by("tail"))
            data_err.add_index("tail")

        if caches is None:
            caches = {}

        # resolve frame
        if frame is not None:
            frame = parse_framelike(frame)

        # initialize each arm
        groups_keys = cast(Column, data.groups.keys)
        arm_names: tuple[str, ...] = tuple(groups_keys["tail"])
        arms: dict[str, StreamArm] = {}
        for k in arm_names:
            arm = StreamArm.from_format(
                data.loc[k],
                origin=origin,
                name=k,
                data_err=None if data_err is None else data_err.loc[k],
                frame=frame,
                _cache=caches.get(k),
                format="astropy.table",
            )
            arms[k] = arm

        return cls(arms, name=name)

    # ===============================================================

    @property
    def data_frame(self) -> BaseCoordinateFrame:
        # all the arms must have the same frame
        return self[self._k0].data_frame

    @property
    def origin(self) -> SkyCoord:
        # all the arms must have the same frame
        return self[self._k0].origin

    @property
    def data_coords(self) -> SkyCoord:
        """The concatenated coordinates of all the arms."""
        if len(self._data) > 1:
            sc = concatenate_coords([arm.data_coords for arm in self.values()])
        else:
            sc = self._v0.data_coords
        return sc

    @property
    def coords(self) -> SkyCoord:
        """The concatenated coordinates of all the arms."""
        if len(self._data) == 1:
            sc = self._v0.coords
        else:
            arm0, arm1 = tuple(self.values())
            sc = concatenate_coords((arm0.coords[::-1], arm1.coords))
        return sc

    @property
    def frame(self) -> BaseCoordinateFrame | None:
        """Return a system-centric frame (or None)."""
        frame = self[self._k0].frame
        return frame

    @property
    def has_distances(self) -> bool:
        return all(arm.has_distances for arm in self.values())

    @property
    def has_kinematics(self) -> bool:
        return all(arm.has_kinematics for arm in self.values())

    # ===============================================================
    # Cleaning Data

    def mask_outliers(self, outlier_method: str | OutlierDetectorBase = "scipyKDTreeLOF", **kwargs: Any) -> None:
        """Detect and label outliers, setting their ``order`` to -1."""
        for arm in self.values():
            arm.mask_outliers(outlier_method, **kwargs)

    # ===============================================================
    # Convenience Methods

    fit_frame = StreamArm.fit_frame

    # ===============================================================
    # Misc

    def __len__(self) -> int:
        return sum(map(len, self.values()))

    def __base_repr__(self, max_lines: int | None = None) -> list[str]:
        rs = super().__base_repr__(max_lines=max_lines)

        # 5) contained streams
        datarepr = (
            f"{name}:\n\t\t"
            + "\n\t\t".join(arm.data._base_repr_(html=False, max_width=None, max_lines=10).split("\n")[1:])
            for name, arm in self.items()
        )
        rs.append("  Streams:\n\t" + "\n\t".join(datarepr))

        return rs

    def __repr__(self) -> str:
        return StreamBase.__repr__(self)


@get_frame.register
def _get_frame_stream(stream: Stream, /) -> BaseCoordinateFrame:
    if stream.frame is None:
        raise ValueError("need to fit a frame")
    return stream.frame
