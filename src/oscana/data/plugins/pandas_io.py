"""\
oscana / data / plugins / pandas_io.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

The Pandas-based input/output strategy.

"""

from __future__ import annotations

from typing import List, Dict, Set, Callable
from typing import Any, Union, Optional
from typing import TYPE_CHECKING, TypeAlias, TypedDict

__all__ = ["PandasIO"]

import logging
from pathlib import Path

import json
import h5py
import uproot

import numpy as np
import pandas as pd

import numpy.typing as npt

from ...logger import _error, _warn
from ..io_base import (
    _DataIOStrategy,
    _SupportedCompressionType,
    _H5DataTypeConverter,
    ALL_HDF5_VARIABLES,
    H5_DATA_BRANCH_NAME,
    H5_CUTS_BRANCH_NAME,
    H5_META_BRANCH_NAME,
)
from ..t_metadata import TransformMetadata
from ..f_metadata import FileMetadata
from ...utils import OscanaError, _get_dir_from_env

if TYPE_CHECKING:
    from ..data_handler import DataHandler


# =============================== [ Logging  ] =============================== #

logger = logging.getLogger("Root")


# ============================== [ Constants  ] ============================== #

ERROR_IN_WARN_FORMAT = "[E//{error}]"  # ~ so I can Ctrl+F for it :)


class _FileLoaderResult(TypedDict):
    """\
    Returned by file loaders.
    """

    mini_data_df: pd.DataFrame
    mini_cuts_df: pd.DataFrame
    file_metadata: List[FileMetadata]
    transform_metadata: TransformMetadata


_FileLoaderFuncType: TypeAlias = Callable[[Path, List[str]], _FileLoaderResult]

# =============================== [ Helpers  ] =============================== #


def _resolve_file_directory(file: str) -> Path:
    """\
    [ Internal ] Get the file directory from enviornment variables or from
    a path.
    """
    try:  # ~ is it in the environment variables?...
        file_path = _get_dir_from_env(file=file)
        logger.debug(f"Retrived '{file}' from the environment variables.")
        return file_path
    except OscanaError:
        logger.debug(
            f"Failed to find '{file}' in the environment variables "
            "- trying it as a path instead."
        )

    file_path = Path(file)  # ~ ... if not, then it must be a path.

    if not file_path.is_file():
        _error(OscanaError, f"File '{file!s}' does not exist!", logger)

    logger.debug(f"Found file '{file!s}' at '{file_path!s}'.")

    return file_path


def _post_file_loader_checks(
    file_name: str,
    result: _FileLoaderResult,
    parent_data_columns: List[str],
    parent_cuts_columns: List[str],
    current_t_metadata: TransformMetadata,
) -> None:
    """\
    [ Internal ] Perform checks after the file loader has done its job.
    """
    # Check transform metadata...
    if current_t_metadata != result["transform_metadata"]:
        _error(
            OscanaError,
            "All files must have the same transforms applied! The transforms "
            f"for '{file_name}' are different from the previous files.",
            logger,
        )

    # Check columns (so there are no future issues with `concat`).
    if len(parent_data_columns) and (
        set(result["mini_data_df"].columns) != set(parent_data_columns)
    ):
        _error(
            OscanaError,
            "All files must have the same columns! The columns for "
            f"'{file_name}' are different from the previous files.",
            logger,
        )

    if len(parent_cuts_columns) and (
        set(result["mini_cuts_df"].columns) != set(parent_cuts_columns)
    ):
        _error(
            OscanaError,
            "All files must have the same cut columns! The cut "
            f"columns for '{file_name}' are different from the previous "
            "files.",
            logger,
        )


# ROOT files


def _create_mini_df_from_uproot(
    uproot_file: uproot.ReadOnlyDirectory, variables: List[str]
) -> pd.DataFrame:
    """\
    [ Internal ] Load a "mini" `DataFrame` from a sinlge Uproot file.
    """
    file_data: dict[str, npt.NDArray] = {}

    for full_variable_name in variables:
        logger.debug(f"Extracting variable '{full_variable_name!s}'...")

        variable_name = full_variable_name.split("/")[-1]

        file_data[variable_name] = uproot_file[
            full_variable_name
        ].array(  # pyright: ignore[reportAttributeAccessIssue]
            library="np"
        )

    return pd.DataFrame(file_data)


def _root_file_loader(
    file_path: Path, variables: List[str]
) -> _FileLoaderResult:
    """\
    [ Internal ] Load data from a ROOT file using Uproot.
    """
    uproot_file = uproot.open(file_path)
    logger.debug(f"Opened {file_path!s} using Uproot.")

    file_metadata = FileMetadata.from_sntp(
        file_name=file_path.name, file=uproot_file
    )  # ~ will throw if the file is not actually an SNTP file

    mini_data_df = _create_mini_df_from_uproot(
        uproot_file=uproot_file,  # pyright: ignore[reportArgumentType]
        variables=variables,
    )
    uproot_file.close()  # pyright: ignore[reportAttributeAccessIssue]

    mini_cuts_df = pd.DataFrame()  # ~ will always be empty for ROOT files

    return _FileLoaderResult(
        mini_data_df=mini_data_df,
        mini_cuts_df=mini_cuts_df,
        file_metadata=[file_metadata],
        transform_metadata=TransformMetadata(),
    )


# HDF5 files


def _build_compression_kwargs(
    compression: _SupportedCompressionType, compression_level: int
) -> dict[str, Any]:
    """\
    [ Internal ] Build the compression keyword arguments for h5py.
    """
    if compression is None:
        return {}

    if compression not in ["gzip", "lzf"]:
        _error(
            OscanaError,
            f"Compression '{compression}' is not supported! Use 'gzip' or "
            "'lzf'.",
            logger,
        )

    if compression == "gzip":
        if not (0 <= compression_level <= 9):
            _error(
                OscanaError,
                f"Compression level '{compression_level}' is not valid! Use "
                "an integer between 0 and 9.",
                logger,
            )

        return {
            "compression": compression,
            "compression_opts": compression_level,
        }

    return {"compression": compression}


def _is_jagged_array(data: pd.Series) -> bool:
    """\
    [ Internal ]Check if the data is a jagged array (i.e., a list of lists).
    """
    if data.empty:
        return False  # ~ AI told me to add this check. It should never happen
        #   though - I think...

    # Note: I am only checking the first element here. So, I assume that all
    #       elements have the same type! This is reasonable because that's how
    #       our data is... hopefully.
    return isinstance(data.iloc[0], np.ndarray) and (data.dtype == "object")


def _fill_h5_branch_from_df(
    group: h5py.Group,
    df: pd.DataFrame,
    data_type_converter: Optional[_H5DataTypeConverter],
    compression_kwargs: Dict[str, Any],
) -> None:
    """\
    [ Internal ] Fill a branch in an HDF5 file from a `DataFrame`.
    """
    for column in df.columns:
        data_type: Union[str, h5py.special_dtype] = df[column].dtype

        if _is_jagged_array(data=df[column]):
            data_type = h5py.special_dtype(vlen=df[column].iloc[0].dtype)
        elif data_type_converter is not None:
            data_type = data_type_converter(column, str(data_type))

        group.create_dataset(
            name=column,
            dtype=data_type,
            data=df[column].to_numpy(),
            **compression_kwargs,
        )


def _fill_h5_metadata_branch(
    group: h5py.Group,
    file_metadata: List[FileMetadata],
    transform_metadata: TransformMetadata,
    compression_kwargs: Dict[str, Any],
) -> None:
    """\
    [ Internal ] Fill the metadata branch in an HDF5 file.
    """
    group.create_dataset(
        name="transforms",
        dtype=h5py.string_dtype(encoding="utf-8"),
        data=json.dumps(transform_metadata.to_dict()).encode("utf-8"),
        **compression_kwargs,
    )

    group.create_dataset(
        name="files",
        dtype=h5py.string_dtype(encoding="utf-8"),
        data=json.dumps([fm.to_dict() for fm in file_metadata]).encode("utf-8"),
        **compression_kwargs,
    )


def _read_h5_metadata_branch(
    file: h5py.File,
) -> tuple[List[FileMetadata], TransformMetadata]:
    """\
    [ Internal ] Read the metadata branch in an HDF5 file.
    """
    t_branch = file["meta/transforms"]

    if not isinstance(t_branch, h5py.Dataset):
        _error(OscanaError, "The 'transforms' branch is not a dataset!", logger)

    t_branch_dict = json.loads(t_branch[()].decode("utf-8"))
    t_metadata = TransformMetadata.from_dict(meta_dict=t_branch_dict)

    f_branch = file["meta/files"]

    if not isinstance(f_branch, h5py.Dataset):
        _error(OscanaError, "The 'files' branch is not a dataset!", logger)

    f_branch_dict = json.loads(f_branch[()].decode("utf-8"))
    f_metadata = [FileMetadata.from_dict(meta_dict=f) for f in f_branch_dict]

    return f_metadata, t_metadata


def _create_mini_df_from_h5(
    file: h5py.File, group_name: str, variables: List[str]
) -> pd.DataFrame:
    """\
    [ Internal ] Load a "mini" `DataFrame` from a single HDF5 file.
    """
    file_data: dict[str, npt.NDArray] = {}

    for full_variable_name in variables:
        logger.debug(f"Extracting variable '{full_variable_name!s}'...")

        variable_name = full_variable_name.split("/")[-1]

        h5_branch = file[f"{group_name}/{variable_name}"]

        file_data[variable_name] = np.asarray(
            h5_branch[()]  # pyright: ignore[reportIndexIssue]
        )

    return pd.DataFrame(file_data)


def _hdf5_file_loader(
    file_path: Path, variables: List[str]
) -> _FileLoaderResult:
    """\
    [ Internal ] Load data from a HDF5 file.
    """
    with h5py.File(file_path, "r") as h5_file:
        logger.debug(f"Opened {file_path!s} using `h5py`.")

        if ALL_HDF5_VARIABLES in variables:
            variables = list(
                h5_file[
                    "data"
                ].keys()  # pyright: ignore[reportAttributeAccessIssue]
            )

        cut_variables = list(
            h5_file[
                "cuts"
            ].keys()  # pyright: ignore[reportAttributeAccessIssue]
        )

        file_metadata, transform_metadata = _read_h5_metadata_branch(
            file=h5_file
        )

        mini_data_df = _create_mini_df_from_h5(
            file=h5_file, group_name=H5_DATA_BRANCH_NAME, variables=variables
        )
        mini_cuts_df = _create_mini_df_from_h5(
            file=h5_file,
            group_name=H5_CUTS_BRANCH_NAME,
            variables=cut_variables,
        )

    return _FileLoaderResult(
        mini_data_df=mini_data_df,
        mini_cuts_df=mini_cuts_df,
        file_metadata=file_metadata,
        transform_metadata=transform_metadata,
    )


# ============================= [ IO Strategy  ] ============================= #


class PandasIO(_DataIOStrategy[pd.DataFrame]):
    def __init__(self, parent: "DataHandler[pd.DataFrame]") -> None:
        """\
        Pandas IO Strategy.
        """
        super().__init__(parent=parent)

    # Helpers

    def _update_parent(
        self,
        cache: Set[str],
        mini_data_dfs: List[pd.DataFrame],
        mini_cuts_dfs: List[pd.DataFrame],
        transform_metadata: TransformMetadata,
        file_metadata: List[FileMetadata],
    ) -> None:
        """\
        [ Internal ] Update the parent `DataHandler` with the loaded data and 
        metadata.
        """
        try:
            # Data
            self._parent._data_table = pd.concat(
                [self._parent._data_table, *mini_data_dfs], ignore_index=True
            )

            # Cuts
            if self._parent.has_cuts_table:
                self._parent._cuts_table = pd.concat(
                    [self._parent._cuts_table, *mini_cuts_dfs],
                    ignore_index=True,
                )
            else:  # ~ if no cuts table, then just add cuts to data table
                self._parent._data_table = pd.concat(
                    [self._parent._data_table, *mini_cuts_dfs], axis=1
                )
        except Exception as e:
            # Cache - remove all the files in that we failed to load
            self._cache = self._cache.difference(cache)

            _error(
                OscanaError,
                f"An error occurred while updating the `DataHandler` "
                f"{ERROR_IN_WARN_FORMAT.format(error=e)}!",
                logger=logger,
            )
        else:
            # Metadata - only do it if there are no errors when joining DFs
            self._parent._t_metadata = transform_metadata
            self._parent._f_metadata.extend(file_metadata)

    def _load_from_files(
        self,
        user_files: List[Union[str, Path]],
        file_loader_func: _FileLoaderFuncType,
    ) -> None:
        """\
        [ Internal ] Load data from a list of files using the given file loader
        function.
        """
        exceptions_: List[Exception] = []

        cache_proxy: Set[str] = self._cache.copy()  # ~ a proxy
        mini_data_dfs: List[pd.DataFrame] = []
        mini_cuts_dfs: List[pd.DataFrame] = []
        f_metadata_list: List[FileMetadata] = []
        t_metadata = self._parent._t_metadata  # ~ a proxy

        # Note: Variable naming here is not the best. `name` refers to the user
        #       -provided file name (in ".env" or a path), while `path` refers
        #       to the actual expanded file path.

        for name in user_files:
            name = str(name)  # ~ the name should be a string for the cache

            if name in self._cache:
                logger.warning(
                    f"File '{name}' has already been loaded! Skipping..."
                )
                continue

            try:
                path = _resolve_file_directory(file=name)

                logger.debug(f"Trying to load data from '{name}'...")
                result = file_loader_func(path, self._parent._variables)

                if not len(self._parent._f_metadata):
                    # Only update the transform metadata like this if there
                    # are no file loaded yet.
                    t_metadata = result["transform_metadata"]

                _post_file_loader_checks(
                    file_name=name,
                    result=result,
                    parent_data_columns=self.get_vars_data_table(),
                    parent_cuts_columns=self.get_vars_cuts_table(),
                    current_t_metadata=t_metadata,
                )

            except Exception as e:
                # ~ this ensures that the program does not stop if one of the
                #   files fail to load - more useful in interactive sessions
                _warn(
                    RuntimeWarning,
                    f"An error occurred while loading data from {name} "
                    f"{ERROR_IN_WARN_FORMAT.format(error=e)}!",
                    logger=logger,
                )
                exceptions_.append(e)
            else:
                self._cache.add(name)  # ~ this is important!
                cache_proxy.add(name)
                mini_data_dfs.append(result["mini_data_df"])
                mini_cuts_dfs.append(result["mini_cuts_df"])
                f_metadata_list.extend(result["file_metadata"])

        self._update_parent(
            cache=cache_proxy,
            mini_data_dfs=mini_data_dfs,
            mini_cuts_dfs=mini_cuts_dfs,
            transform_metadata=t_metadata,
            file_metadata=f_metadata_list,
        )

        if len(exceptions_):
            _warn(
                RuntimeWarning,
                f"Loading finished with {len(exceptions_)} errors. Some files "
                "may have failed to load. (See the above exceptions.)",
                logger=logger,
            )

    # Overrides

    def _init_data_table(self) -> pd.DataFrame:
        """\
        [ Internal ] Initialise the data table.
        """
        return pd.DataFrame()

    def _init_cuts_table(self) -> pd.DataFrame:
        """\
        [ Internal ] Initialise the cuts table.
        """
        return pd.DataFrame()

    def from_sntp(self, files: List[Union[str, Path]]) -> None:
        """\
        Load data from MINOS SNTP ROOT files.

        Parameters
        ----------
        files : List[Union[str, Path]]
            List of files.
        """
        # Note: Duplicate files are taken care of by the `_cache` checks.
        self._load_from_files(
            user_files=files, file_loader_func=_root_file_loader
        )

    def from_udst(self, files: List[Union[str, Path]]) -> None:
        """\
        Load data from MINOS uDST (micro-DST) ROOT files.

        Parameters
        ----------
        files : List[Union[str, Path]]
            List of files.
        """
        raise NotImplementedError(
            "Loading from uDST files is not yet implemented!"
        )

    def from_hdf5(self, files: List[Union[str, Path]]) -> None:
        """\
        Load data from HDF5 files.
        
        Parameters
        ----------
        files : List[Union[str, Path]]
            List of files.
        """
        self._load_from_files(
            user_files=files, file_loader_func=_hdf5_file_loader
        )

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
        # Check file path...
        out_file = Path(out_file)

        if not (out_file.is_file and (out_file.suffix == ".h5")):
            _error(
                OscanaError,
                f"The out file '{out_file!s}' is not an HDF5 file! Provide a "
                "path to a file with the '.h5' extension.",
                logger,
            )

        if ensure_parent_dir:
            out_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to HDF5...
        compression_kwargs = _build_compression_kwargs(
            compression=compression, compression_level=compression_level
        )

        with h5py.File(out_file, "w") as h5_file:
            data_branch = h5_file.create_group(name=H5_DATA_BRANCH_NAME)
            _fill_h5_branch_from_df(
                group=data_branch,
                df=self._parent._data_table,
                data_type_converter=data_type_converter,
                compression_kwargs=compression_kwargs,
            )

            cuts_branch = h5_file.create_group(name=H5_CUTS_BRANCH_NAME)
            if self._parent.has_cuts_table:
                _fill_h5_branch_from_df(
                    group=cuts_branch,
                    df=self._parent._cuts_table,
                    data_type_converter=data_type_converter,
                    compression_kwargs=compression_kwargs,
                )

            meta_branch = h5_file.create_group(name=H5_META_BRANCH_NAME)
            _fill_h5_metadata_branch(
                group=meta_branch,
                file_metadata=self._parent._f_metadata,
                transform_metadata=self._parent._t_metadata,
                compression_kwargs=compression_kwargs,
            )

    def get_n_rows_data_table(self) -> int:
        """\
        Get the length of the data table.
        """
        return len(self._parent._data_table)

    def get_n_rows_cuts_table(self) -> int:
        """\
        Get the length of the cuts table.

        Returns
        -------
        int
            The length of the cuts table.
        """
        if self._parent.has_cuts_table:
            return len(self._parent._cuts_table)

        return 0

    def get_n_vars_data_table(self) -> int:
        """\
        Get the number of variables in the data table.
        """
        return len(self._parent._data_table.columns)

    def get_n_vars_cuts_table(self) -> int:
        """\
        Get the number of variables in the cuts table.
        """
        if self._parent.has_cuts_table:
            return len(self._parent._cuts_table.columns)

        return 0

    def get_vars_data_table(self) -> List[str]:
        """\
        Get the list of variable names in the data table.
        """
        return list(self._parent._data_table.columns)

    def get_vars_cuts_table(self) -> List[str]:
        """\
        Get the list of variable names in the cuts table.
        """
        if self._parent.has_cuts_table:
            return list(self._parent._cuts_table.columns)

        return []
