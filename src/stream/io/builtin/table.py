# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import annotations

# STDLIB
import copy
import inspect
from typing import TYPE_CHECKING, Any, cast

# THIRD PARTY
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Table

# LOCAL
from stream.io.core import convert_registry
from stream.io.normalize import StreamArmDataNormalizer
from stream.stream.core import StreamArm
from stream.utils.coord_utils import parse_framelike

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.coordinates import BaseCoordinateFrame

__all__: list[str] = []


# ===================================================================


def stream_arm_from_table(
    table: Table,
    /,
    data_err: QTable | None = None,
    *,
    name: str | None = None,
    frame: BaseCoordinateFrame | None,
    origin: SkyCoord | None = None,
    Stream: type[StreamArm] | None = None,
    _cache: dict[str, Any] | None = None,
) -> StreamArm:
    table_meta = cast("dict[str, Any]", copy.copy(table.meta))
    # Stream class
    if Stream is None:
        Stream = table_meta.pop("Stream")
    if not (inspect.isclass(Stream) and issubclass(Stream, StreamArm)):
        raise TypeError

    # Cache:
    # it's not a passable parameter. It's specific to the table.
    # TODO! should it be meta[...] or meta[cache[]]
    meta_cache = table_meta.pop("cache", None)
    if _cache is None:
        _cache = meta_cache
    else:
        raise ValueError("conflicting caches")
    cache: dict[str, Any] = {} if _cache is None else _cache

    # Name:
    meta_name = table_meta.pop("name", None)
    name = meta_name if name is None else name

    # Frame (CoordinateFrame | None)
    meta_frame = table_meta.pop("frame", None)
    cache_frame = cache.get("frame")
    if frame is None:
        frame = cache_frame if meta_frame is None else meta_frame
    elif meta_frame is not None and frame != meta_frame:
        raise ValueError("frame != one in table meta")
    elif cache_frame is not None and frame != cache_frame:
        raise ValueError("frame != one in table meta")
    frame = parse_framelike(frame) if frame is not None else None

    # Origin (SkyCoord) can also be on table meta
    meta_origin = table_meta.pop("origin", None)
    if origin is None:
        origin = meta_origin
    elif meta_origin is not None and origin != meta_origin:
        raise ValueError("origin != one in table meta")
    if origin is None:
        raise ValueError("need arg origin or origin in table meta")
    elif not isinstance(origin, SkyCoord):
        raise TypeError

    # Data (Table)
    # TODO! offer different normalizers
    data: QTable
    data = StreamArmDataNormalizer(frame)(table, data_err)

    stream = Stream(data, origin=origin, frame=frame, name=name, prior_cache=cache)
    return stream


def table_identify(origin: str, format: str | None, /, *args: Any, **kwargs: Any) -> bool:
    """Identify if object uses the Table format.

    Returns
    -------
    bool
    """
    itis = False
    if origin == "read":
        itis = isinstance(args[1], Table) and (format in (None, "astropy.table"))
    return itis


# ===================================================================
# Register

register_StreamArm_from_format = {
    "registry": convert_registry,
    "data_class": StreamArm,
    "func": stream_arm_from_table,
    "identify": table_identify,
}
