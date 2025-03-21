"""\
oscana / data / io_base.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = []

from typing import TYPE_CHECKING, TypeAlias, TypeVar, Callable, Generic

import logging
from pathlib import Path
from abc import ABC, abstractmethod

from .f_metadata import FileMetadata
from .t_metadata import TransformMetadata
from ..utils import _error

if TYPE_CHECKING:
    from .data_handler import DataHandler

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

# =============================== [ Logging  ] =============================== #

logger = logging.getLogger("Root")

# =========================== [ Helper Functions ] =========================== #


def _get_non_cache_files(cache: list[str], files: list[str]) -> list[str]:
    """\
    [ Internal ]

    Get the files that are not in the cache.

    Parameters
    ----------
    cache : list[str]
        List of files that are already in the cache.

    files : list[str]
        List of files to check against the cache.

    Returns
    -------
    list[str]
        List of files that are not in the cache.
    """
    non_cache_files = []

    for file in files:
        if file in cache:
            logger.info(f"Skipping '{file}' as it is already in the cache.")
            continue

        non_cache_files.append(file)

    cache.extend(non_cache_files)

    return non_cache_files


# =========================== [ Data IO Strategy ] =========================== #


class DataIOStrategy(ABC, Generic[T]):  # Can't use the cool 3.12 syntax :(

    _sntp_loader: LoaderFuncType[T]
    _udst_loader: LoaderFuncType[T]
    _hdf5_loader: LoaderFuncType[T]

    _hdf5_writer: WriterFuncType[T]

    def __init__(self, parent: DataHandler) -> None:
        self._parent: DataHandler = parent
        self._cache: list[str] = []

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
    def _from_sntp(self, files: list[str]) -> None:
        pass

    @abstractmethod
    def _from_udst(self, files: list[str]) -> None:
        pass

    @abstractmethod
    def _from_hdf5(self, files: list[str]) -> None:
        pass

    def from_sntp(self, files: list[str]) -> None:
        """\
        Load data from SNTP ROOT files.

        Parameters
        ----------
        files : list[str]
            List of names of the SNTP ROOT files.  

        Notes
        -----
        The names of the files should be from the environment variables (i.e. 
        the '.env' file).
        """
        files = _get_non_cache_files(self._cache, files)

        if not files:
            return

        return self._from_sntp(files=files)

    def from_udst(self, files: list[str]) -> None:
        """\
        Load data from uDST ROOT files.

        Parameters
        ----------
        files : list[str]
            List of names of the uDST ROOT files.

        Notes
        -----
        The names of the files should be from the environment variables (i.e.
        the '.env' file).
        """
        files = _get_non_cache_files(self._cache, files)

        if not files:
            return

        return self._from_udst(files=files)

    def from_hdf5(self, files: list[str]) -> None:
        """\
        Load data from HDF5 files.

        Parameters
        ----------
        files : list[str]
            List of names of the HDF5 files.

        Notes
        -----
        The names of the files should be from the environment variables (i.e.
        the '.env' file).
        """
        files = _get_non_cache_files(self._cache, files)

        if not files:
            return

        return self._from_hdf5(files=files)

    def to_hdf5(self, file: str | Path) -> None:
        _error(
            NotImplementedError,
            "The HDF5 writer is not implemented yet!",
            logger,
        )

    def __str__(self) -> str:
        return (
            f"Oscana.{self.__class__.__name__}("
            f"parent={self._parent.__class__.__name__})"
        )

    def __repr__(self) -> str:
        return str(self)
