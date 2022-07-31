"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    TypeVar,
    ValuesView,
)

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import BaseCoordinateFrame, RadialDifferential, SkyCoord
from astropy.utils.misc import indent

# LOCAL
from stream.utils.descriptors.attribute import Attribute
from stream.utils.descriptors.cache import CacheProperty

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.table import QTable


__all__ = ["StreamBase", "CollectionBase"]

##############################################################################
# PARAMETERS

_ABC_MSG = "Can't instantiate abstract class {} with abstract method {}"
# Error message for ABCs.
# ABCs only prevent a subclass from being defined if it doesn't override the
# necessary methods. ABCs do not prevent empty methods from being called. This
# message is for errors in abstract methods.

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class Flags:
    minPmemb: u.Quantity = u.Quantity(80, unit=u.percent)
    table_repr_max_lines: int = 10


##############################################################################


@dataclass(frozen=True)
class StreamBase:
    """Abstract base class for stream arms, and collections thereof.

    Streams must define the following attributes / properties.

    Attributes
    ----------
    data : `astropy.table.QTable`
        The Stream data.
    origin : `astropy.coordinates.SkyCoord`
        The origin of the stream.
    name : str
        Name of the stream.
    """

    cache = CacheProperty["StreamBase"]()
    flags = Attribute(Flags(minPmemb=u.Quantity(90, unit=u.percent), table_repr_max_lines=10), attrs_loc="__dict__")

    # ===============================================================
    # Initializatino

    # TODO! py3.10 fixes the problems of ordering in subclasses
    # data: QTable
    # """The stream data table."""
    # name: str | None
    # """The name of the stream."""
    # prior_cache: InitVar[dict] | None = None
    # def __post_init__(self, prior_cache: dict | None) -> None:
    #     self.cache = prior_cache

    # this is included only for type hinting
    def __init__(self) -> None:
        self._cache: dict[str, Any]
        self.data: QTable
        self.origin: SkyCoord
        self.name: str | None

    # ===============================================================
    # Flags

    @property
    def has_distances(self) -> bool:
        """Whether the data has distances or is on-sky."""
        data_onsky = self.data["coord"].spherical.distance.unit.physical_type == "dimensionless"
        origin_onsky = self.origin.spherical.distance.unit.physical_type == "dimensionless"
        onsky: bool = data_onsky and origin_onsky
        return not onsky

    @property
    def has_kinematics(self) -> bool:
        """Return `True` if ``.coords`` has distance information."""
        has_vs = "s" in self.data["coord"].data.differentials

        # For now can't do RadialDifferential
        if has_vs:
            has_vs &= not isinstance(self.data["coord"].data.differentials["s"], RadialDifferential)

        return has_vs

    # ===============================================================

    # @property
    # @abstractmethod
    # def origin(self) -> SkyCoord:
    #     """The origin of the stream."""
    #     raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "origin"))

    @property
    # @abstractmethod
    def data_frame(self) -> BaseCoordinateFrame:
        """The frame of the coordinates in ``data``."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "data_frame"))

    @property
    # @abstractmethod
    def data_coords(self) -> SkyCoord:
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "data_coords"))

    @property
    # @abstractmethod
    def frame(self) -> BaseCoordinateFrame | None:
        """The stream data."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "frame"))

    @property
    # @abstractmethod
    def coords(self) -> SkyCoord:
        """Coordinates."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "coords"))

    @property
    def full_name(self) -> str | None:
        """The name of the stream."""
        return self.name

    # ===============================================================

    def __base_repr__(self, max_lines: int | None = None) -> list[str]:
        rs = []

        # 0) header (standard repr)
        header: str = object.__repr__(self)
        rs.append(header)

        # 1) name
        name = str(self.full_name)
        rs.append("  Name: " + name)

        # 2) frame
        frame: str = repr(self.frame)
        r = "  Frame:"
        r += ("\n" + indent(frame)) if "\n" in frame else (" " + frame)
        rs.append(r)

        # 3) Origin
        origin: str = repr(self.origin.transform_to(self.data_frame))
        r = "  Origin:"
        r += ("\n" + indent(origin)) if "\n" in origin else (" " + origin)
        rs.append(r)

        # 4) data frame
        data_frame: str = repr(self.data_frame)
        r = "  Data Frame:"
        r += ("\n" + indent(data_frame)) if "\n" in data_frame else (" " + data_frame)
        rs.append(r)

        return rs

    def __repr__(self) -> str:
        s: str = "\n".join(self.__base_repr__(max_lines=self.flags.table_repr_max_lines))
        return s

    def __len__(self) -> int:
        return len(self.data)


##############################################################################


class CollectionBase(Mapping[str, V]):
    """Base class for a homogenous, keyed collection of objects.

    Parameters
    ----------
    data : dict[str, V] or None, optional
        Mapping of the data for the collection.
        If `None` (default) the collection is empty.
    name : str or None, optional keyword-only
        The name of the collection
    **kwargs : V, optional
        Further entries for the collection.
    """

    __slots__ = ("_data", "name")

    def __init__(self, data: dict[str, V] | None = None, /, *, name: str | None = None, **kwargs: V) -> None:
        d = data if data is not None else {}
        d.update(kwargs)

        self._data: dict[str, V]
        object.__setattr__(self, "_data", d)

        self.name: str | None
        object.__setattr__(self, "name", name)

    @property
    def data(self) -> MappingProxyType[str, V]:
        return MappingProxyType(self._data)

    def __getitem__(self, key: str) -> V:
        """Get 'key' from the data."""
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        return self._data.values()

    def items(self) -> ItemsView[str, V]:
        return self._data.items()

    @property
    def _k0(self) -> str:
        return next(iter(self._data.keys()))

    @property
    def _v0(self) -> V:
        return next(iter(self._data.values()))

    def __getattr__(self, key: str) -> MappingProxyType[str, Any]:
        """Map any unkown methods to the contained fields."""
        if hasattr(self._v0, key):
            return MappingProxyType({k: getattr(v, key) for k, v in self.items()})
        raise AttributeError(f"{self.__class__.__name__!r} object has not attribute {key!r}")
