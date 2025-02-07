"""\
oscana / data / plugins / pandas_v1.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = ["PandasIO"]  # Only export the class (any other stuff is ignored)!

from typing import TYPE_CHECKING

import logging
from pathlib import Path

import uproot
import pandas as pd
import numpy.typing as npt

from ..io_base import (
    DataIOStrategy,
    LoaderFuncType,
    WriterFuncType,
    LoadedDataType,
)
from ..t_metadata import TransformMetadata
from ..f_metadata import FileMetadata
from ...utils import (
    OscanaError,
    get_func_lookup,
    _get_dir_from_env,
    _error,
)

if TYPE_CHECKING:
    from ..data_handler import DataHandler


# ================================ [ Logger ] ================================ #

logger = logging.getLogger("Root")

# =============================== [ Helpers  ] =============================== #


def _v1_naive_loader(
    variables: list[str], file: str
) -> tuple[pd.DataFrame, FileMetadata]:
    logger.debug(f"Loading variables from '{file}' using the V1 Naive Loader.")

    file_dir = _get_dir_from_env(file=file)

    uproot_file = uproot.open(file_dir)

    logger.info(f"Opened '{file}' using Uproot.")

    # This is a really crappy way to extract the metadata...
    metadata = FileMetadata.from_sntp(
        file_name=Path(file_dir).name, file=uproot_file
    )

    data_dict: dict[str, npt.NDArray] = {}

    for variable in variables:
        logger.debug(f"Extracting variable '{variable}' from '{file}'...")

        # Note: Not great that we need to specify a base in this way.

        # TODO: Fix this.
        base = variable.split("/")[0]
        key = "/".join(variable.split("/")[1:])

        # Note: I have added some `pyright` comments to suppress annoying
        #       warnings.

        try:
            base_branch = uproot_file[base]
        except uproot.KeyInFileError:
            _error(
                OSError,
                f"Base '{base}' not found in '{file}'!",
                logger,
            )

        try:
            data_dict[key] = base_branch[
                key
            ].arrays(  # pyright: ignore[reportAttributeAccessIssue]
                library="np"
            )[
                key.split("/")[-1]
            ]
        except uproot.KeyInFileError:
            _error(
                OSError,
                f"Variable '{key}' not found in '{file}'!",
                logger,
            )

    uproot_file.close()  # pyright: ignore[reportAttributeAccessIssue]

    logger.info(f"Extracted variables from '{file}'.")

    return pd.DataFrame(data_dict), metadata


# =========================== [ Dynamic Helpers  ] =========================== #


def hlp_20250205_from_sntp(
    variables: list[str], files: list[str]
) -> LoadedDataType[pd.DataFrame]:
    """\
    [ Internal ]
    
    Name: Naïve Loader V1
    """
    exceptions_ = []
    data_list: list[pd.DataFrame] = []
    f_metadata_list: list[FileMetadata] = []

    for file in files:
        try:
            data, f_meta = _v1_naive_loader(variables, file)

            if len(f_metadata_list):
                if f_meta != f_metadata_list[-1]:
                    _error(
                        OscanaError,
                        "All files must have the same metadata!",
                        logger,
                    )

            data_list.append(data)
            f_metadata_list.append(f_meta)
        except Exception as e:
            exceptions_.append(e)

    if len(exceptions_):
        for e in exceptions_:
            if e.__class__.__name__ == "OscanaError":
                continue

            logger.error(
                f"An execption was supressed! {e.__class__.__name__} - {e!s}"
                + ("." if str(e)[-1] != "." or str(e)[-1] != "!" else "")
            )
        _error(
            OscanaError,
            "One or more files failed to load! (See the above exceptions.)",
            logger,
        )

    return (pd.concat(data_list), f_metadata_list, None)


def hlp_20250205_from_udst(
    variables: list[str], files: list[str | Path]
) -> LoadedDataType[pd.DataFrame]:
    """\
    [ Internal ]

    Not Implemented!
    """
    _error(
        NotImplementedError,
        "The uDST loader is not implemented yet!",
        logger,
    )


def hlp_20250205_from_hdf5(
    variables: list[str], files: list[str | Path]
) -> LoadedDataType[pd.DataFrame]:
    """\
    [ Internal ]

    Not Implemented!
    """
    _error(
        NotImplementedError,
        "The HDF5 loader is not implemented yet!",
        logger,
    )


def hlp_20250205_to_hdf5(
    data: pd.DataFrame,
    file_metadata: FileMetadata,
    transform_metadata: TransformMetadata,
    file: str | Path,
) -> None:
    """\
    [ Internal ]

    Not Implemented!
    """
    _error(
        NotImplementedError,
        "The HDF5 writer is not implemented yet!",
        logger,
    )


# ============================= [ IO Strategy  ] ============================= #

hlp_lookup = get_func_lookup(globals_=globals(), prefix="hlp_")


class PandasIO(DataIOStrategy[pd.DataFrame]):
    _sntp_loader: LoaderFuncType = hlp_lookup("from_sntp")
    _udst_loader: LoaderFuncType = hlp_lookup("from_udst")
    _hdf5_loader: LoaderFuncType = hlp_lookup("from_hdf5")

    _hdf5_writer: WriterFuncType = hlp_lookup("to_hdf5")

    def __init__(self, parent: DataHandler) -> None:
        """\
        Pandas IO Strategy.

        Parameters
        ----------
        parent : DataHandler
            Parent data handler.
        """
        super().__init__(parent=parent)

    def _init_data_table(self) -> pd.DataFrame:
        """\
        Initialize data table.

        Returns
        -------
        DataFrameType
            Data table.
        """
        return pd.DataFrame()

    def _init_cuts_table(self) -> pd.DataFrame:
        """\
        Initialize cuts table.

        Returns
        -------
        DataFrameType
            Cuts table.
        """
        return pd.DataFrame()

    def from_sntp(self, files: list[str]) -> None:
        # We do not expect any `TransformMetadata` from the SNTP files.
        data, f_meta, _ = PandasIO._sntp_loader(self._parent._variables, files)

        self._parent._data_table = pd.concat([self._parent._data_table, data])
        self._parent._f_metadata.extend(f_meta)

    def from_udst(self, files: list[str]) -> None:
        # We do not expect any `TransformMetadata` from the uDST files.
        data, f_meta, _ = self._udst_loader(self._parent._variables, files)

        self._parent._data_table = pd.concat([self._parent._data_table, data])
        self._parent._f_metadata.extend(f_meta)

    def from_hdf5(self, files: list[str]) -> None:
        data, f_meta, _ = self._hdf5_loader(self._parent._variables, files)

        self._parent._data_table = pd.concat([self._parent._data_table, data])
        self._parent._f_metadata.extend(f_meta)

    def to_hdf5(self, file: str | Path) -> None:
        return self._hdf5_writer(
            self._parent._data_table,
            self._parent._f_metadata,
            self._parent._t_metadata,
            file,
        )
