"""\
oscana / data / io_base.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

The data input/output strategy base class defines the expected interface for all
strategies (loaded from "plugins"). 
"""

from __future__ import annotations

from typing import List, Set, Literal
from typing import Union
from typing import TYPE_CHECKING, TypeAlias, TypeVar, Generic

__all__ = []

from pathlib import Path

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .data_handler import DataHandler

# ============================= [ Type Aliases ] ============================= #

# Generic types for "DataFrame"-like structures.
TCov = TypeVar("TCov", covariant=True)
TCon = TypeVar("TCon", contravariant=True)

_SupportedCompressionType: TypeAlias = Literal["gzip", "lzf", None]

# =========================== [ Data IO Strategy ] =========================== #


class _DataIOStrategy(ABC, Generic[TCov]):  # Can't use the cool 3.12 syntax :(
    """\
    [ Internal ] An abstract base class for data input/output strategies.
    """

    def __init__(self, parent: "DataHandler") -> None:
        """\
        Initialises a `_DataIOStrategy` instance.
        """
        self._parent: "DataHandler" = parent
        self._cache: Set[str] = set()

    # Abstract Methods

    @abstractmethod
    def _init_data_table(self) -> TCov:
        """\
        [ Internal ] Initialise the data table.
        """
        pass

    @abstractmethod
    def _init_cuts_table(self) -> TCov:
        """\
        [ Internal ] Initialise the cuts table.
        """
        pass

    @abstractmethod
    def from_sntp(self, files: List[Union[str, Path]]) -> None:
        """\
        Load data from MINOS SNTP ROOT files.

        Parameters
        ----------
        files : List[Union[str, Path]]
            List of files.
        """
        pass

    @abstractmethod
    def from_udst(self, files: List[Union[str, Path]]) -> None:
        """\
        Load data from MINOS uDST (micro-DST) ROOT files.

        Parameters
        ----------
        files : List[Union[str, Path]]
            List of files.
        """
        pass

    @abstractmethod
    def from_hdf5(self, files: List[Union[str, Path]]) -> None:
        """\
        Load data from HDF5 files.
        
        Parameters
        ----------
        files : List[Union[str, Path]]
            List of files.
        """
        pass

    @abstractmethod
    def to_hdf5(
        self,
        file: Union[str, Path],
        compression: _SupportedCompressionType = None,
    ) -> None:
        """\
        Write everything to an HDF5 file.

        Parameters
        ----------
        file : Union[str, Path]
            The name or path of the HDF5 file to write to.

        compression : _SupportedCompressionType
            The compression algorithm to use. If `None`, no compression is used.
            Default is `None`.

        Notes
        -----
        Compression algorithm "szip" is not supported due to licensing.
        """
        pass

    @abstractmethod
    def get_n_rows_data_table(self) -> int:
        """\
        Get the length of the data table.
        """
        pass

    @abstractmethod
    def get_n_rows_cuts_table(self) -> int:
        """\
        Get the length of the cuts table.

        Returns
        -------
        int
            The length of the cuts table.
        """
        pass

    @abstractmethod
    def get_n_vars_data_table(self) -> int:
        """\
        Get the number of variables in the data table.
        """
        pass

    @abstractmethod
    def get_n_vars_cuts_table(self) -> int:
        """\
        Get the number of variables in the cuts table.
        """
        pass

    # Public Methods

    def get_n_vars(self) -> int:
        """\
        Get total number of variables in the data and cuts table.
        """
        return self.get_n_vars_data_table() + self.get_n_vars_cuts_table()

    # Dunders

    def __str__(self) -> str:
        return f"oscana.{self.__class__.__name__}()"

    def __repr__(self) -> str:
        return str(self)
