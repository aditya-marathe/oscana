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

from typing import List, Sequence, Dict, Set, Literal, Callable
from typing import Union, Optional
from typing import TYPE_CHECKING, TypeAlias, TypeVar, Generic

import numpy.typing as npt

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

_H5DataTypeConverter: TypeAlias = Callable[[str, str], str]
# ^ for a given a column name and it's data type, return the data type to use
#   when writing to HDF5.

_Index: TypeAlias = int | Sequence[int] | npt.NDArray

# ============================== [ Constants  ] ============================== #

H5_DATA_BRANCH_NAME: Literal["data"] = "data"
H5_CUTS_BRANCH_NAME: Literal["cuts"] = "cuts"
H5_META_BRANCH_NAME: Literal["meta"] = "meta"

# =========================== [ Data IO Strategy ] =========================== #


class _DataIOStrategy(ABC, Generic[TCov]):
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
        out_file: Union[str, Path],
        data_type_converter: Optional[_H5DataTypeConverter] = None,
        compression: _SupportedCompressionType = None,
        compression_level: int = 6,
        ensure_parent_dir: bool = False,
    ) -> None:
        """\
        Write everything to an HDF5 file.

        Parameters
        ----------
        out_file : Union[str, Path]
            The name or path of the HDF5 file to write to.

        data_type_converter : Optional[_H5DataTypeConverter]
            A function that takes a column name and its data type and returns 
            the data type to use for HDF5. If `None`, nothing is done.
            Default is `None`.

        compression : _SupportedCompressionType
            The compression algorithm to use. If `None`, no compression is used.
            Default is `None`.

        compression_level : int
            The level of compression to use for "gzip" (from 0 to 9). Defaults
            to 6.

        ensure_parent_dir : bool
            If `True`, the parent directory of the output file will be created
            if it does not exist. Default is `False`.

        Notes
        -----
        Compression algorithm "szip" is not supported due to licensing. The out
        file should have the following "branches": "data", "cuts", and "meta".
        """
        pass

    @abstractmethod
    def get_column(
        self,
        name: str,
        indices: Optional[_Index] = None,
        from_cuts: bool = False,
    ) -> npt.NDArray:
        """\
        Get a column from the data or cuts table.

        Parameters
        ----------
        name : str
            The name of the column to get.

        indices : Optional[_Index]
            The indices of the rows to get. If `None`, all rows are returned.

        from_cuts : bool
            Whether to get the column from the cuts table. Defaults to `False`.

        Returns
        -------
        npt.NDArray
            The column values.
        """
        pass

    @abstractmethod
    def get_columns(
        self,
        names: List[str],
        indices: Optional[_Index] = None,
        from_cuts: bool = False,
    ) -> Dict[str, npt.NDArray]:
        """\
        Get a dictionary of columns from the data or cuts table.

        Parameters
        ----------
        names : List[str]
            The names of the columns to get.

        indices : Optional[_Index]
            The indices of the rows to get. If `None`, all rows are returned.

        from_cuts : bool
            Whether to get the columns from the cuts table. Defaults to `False`.

        Returns
        -------
        Dict[str, npt.NDArray]
            A dictionary mapping column names to their values.
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

    @abstractmethod
    def get_vars_data_table(self) -> List[str]:
        """\
        Get the list of variable names in the data table.
        """
        pass

    @abstractmethod
    def get_vars_cuts_table(self) -> List[str]:
        """\
        Get the list of variable names in the cuts table.
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
