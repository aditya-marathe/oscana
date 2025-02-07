"""\
oscana / data / io_base.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = []

from typing import TYPE_CHECKING, TypeVar, TypeAlias, Callable, Generic

from pathlib import Path
from abc import ABC, abstractmethod

from .f_metadata import FileMetadata
from .t_metadata import TransformMetadata

if TYPE_CHECKING:
    from .data_handler import DataHandler  # Guard against circular imports.

# ============================= [ Type Aliases ] ============================= #

T = TypeVar("T")  # Generic type for "DataFrame"-like structures.

# Note: In Python 3.12, you would want to do this instead:
#       ```python
#       type LoaderFuncType[T] = Callable[...]
#       ```
#       But, alas, we are stuck with stinky & old Python 3.10 for now.

LoaderFuncType: TypeAlias = Callable[
    [list[str], list[str]],
    tuple[T, list[FileMetadata], TransformMetadata | None],
]
WriterFuncType: TypeAlias = Callable[
    [T, list[FileMetadata], TransformMetadata, str | Path], None
]
LoadedDataType: TypeAlias = tuple[
    T, list[FileMetadata], TransformMetadata | None
]


# =========================== [ Data IO Strategy ] =========================== #


class DataIOStrategy(ABC, Generic[T]):  # Can't use the cool 3.12 syntax :(

    _sntp_loader: LoaderFuncType[T]
    _udst_loader: LoaderFuncType[T]
    _hdf5_loader: LoaderFuncType[T]

    _hdf5_writer: WriterFuncType[T]

    def __init__(self, parent: DataHandler) -> None:
        self._parent: DataHandler = parent

    @abstractmethod
    def _init_data_table(self) -> T:
        pass

    @abstractmethod
    def _init_cuts_table(self) -> T:
        pass

    def _get_strategy_info(self) -> dict[str, str]:
        """\
        [ Internal ]
        
        Get the IO strategy information for all "loaders" and "writers".
        
        Returns
        -------
        dict[str, str]
            Dictionary containing the IO strategy information.
        """
        doc = lambda x: x.__doc__.split("\n")[2][4:] if x.__doc__ else "???"
        get = lambda x: doc(x)[6:] if doc(x).startswith("Name: ") else "???"
        return {
            "SNTP Loader": get(self._sntp_loader),
            "uDST Loader": get(self._udst_loader),
            "HDF5 Loader": get(self._hdf5_loader),
            "HDF5 Writer": get(self._hdf5_writer),
        }

    @abstractmethod
    def from_sntp(self, files: list[str]) -> None:
        pass

    @abstractmethod
    def from_udst(self, files: list[str]) -> None:
        pass

    @abstractmethod
    def from_hdf5(self, files: list[str]) -> None:
        pass

    @abstractmethod
    def to_hdf5(self, file: str | Path) -> None:
        pass

    def __str__(self) -> str:
        return (
            f"Oscana.{self.__class__.__name__}("
            f"parent={self._parent.__class__.__name__})"
        )

    def __repr__(self) -> str:
        return str(self)
