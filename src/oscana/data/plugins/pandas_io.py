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
from typing_extensions import override

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

    file_path = Path(file).expanduser().resolve()

    if not file_path.is_file():
        _error(OscanaError, f"File '{file!s}' does not exist!", logger)

    logger.debug(f"Found file '{file!s}' at '{file_path!s}'.")

    return file_path


def _reorder_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """\
    [ Internal ] Order the columns of a `DataFrame` in a certain way.
    """
    if list(df.columns) == columns:
        return df

    return df[columns]


def _post_file_loader_checks(
    file_name: str,
    result: _FileLoaderResult,
    data_vars_proxy: List[str],
    cuts_vars_proxy: List[str],
    current_t_metadata: TransformMetadata,
    has_cuts_table: bool,
) -> None:
    """\
    [ Internal ] Perform checks after the file loader has done its job.
    """
    # A file that carries cuts can only be loaded into a `DataHandler` that was
    # set up with a cuts table - otherwise those cuts would be silently thrown
    # away.
    if not has_cuts_table and len(result["mini_cuts_df"].columns):
        _error(
            OscanaError,
            f"The file '{file_name}' was saved with cuts "
            f"{sorted(result['mini_cuts_df'].columns)}, but this "
            "`DataHandler` has no cuts table! Re-create it with "
            "`make_cut_bool_table=True`.",
            logger,
        )

    # Check for data/cut name collisions. Now, this should never happen but, oh
    # well, it does not hurt to check.
    duplicate_vars = set(result["mini_data_df"].columns) & set(
        result["mini_cuts_df"].columns
    )
    if duplicate_vars:
        _error(
            OscanaError,
            f"Variable(s) {sorted(duplicate_vars)} in '{file_name}' are "
            "used as both a data and a cut variable!",
            logger,
        )

    # Check transform metadata...
    if current_t_metadata != result["transform_metadata"]:
        _error(
            OscanaError,
            "All files must have the same transforms applied! The transforms "
            f"for '{file_name}' are different from the previous files.",
            logger,
        )

    # Check columns (so there are no future issues with `concat`). Note: the
    # comparison is order-insensitive, since `_reorder_columns` puts every
    # frame into the order of the proxy before we concatenate anything.
    if len(data_vars_proxy) == 0:
        data_vars_proxy.extend(result["mini_data_df"].columns)

    if set(result["mini_data_df"].columns) != set(data_vars_proxy):
        _error(
            OscanaError,
            "All files must have the same columns! The columns for "
            f"'{file_name}' are different from the previous files.",
            logger,
        )

    if len(cuts_vars_proxy) == 0:
        cuts_vars_proxy.extend(result["mini_cuts_df"].columns)

    if set(result["mini_cuts_df"].columns) != set(cuts_vars_proxy):
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
    with uproot.open(
        file_path
    ) as uproot_file:  # pyright: ignore[reportGeneralTypeIssues]
        logger.debug(f"Opened {file_path!s} using Uproot.")

        file_metadata = FileMetadata.from_sntp(
            file_name=file_path.name, file=uproot_file
        )  # ~ will throw if the file is not actually an SNTP file

        mini_data_df = _create_mini_df_from_uproot(
            uproot_file=uproot_file,  # pyright: ignore[reportArgumentType]
            variables=variables,
        )

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
        data_type = df[column].dtype

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


def _check_h5_group_has_variables(
    file: h5py.File, group_name: str, variables: List[str]
) -> None:
    """\
    [ Internal ] Check if variables exist.
    """
    group = file[group_name]

    if not isinstance(group, h5py.Group):
        _error(
            OscanaError,
            f"The '{group_name}' branch is not a group!",
            logger,
        )

    available_variables = set(group.keys())

    missing_variables = sorted(
        {
            variable.split("/")[-1]
            for variable in variables
            if variable.split("/")[-1] not in available_variables
        }
    )

    if missing_variables:
        _error(
            OscanaError,
            f"Variable(s) {missing_variables} were not found in the "
            f"'{group_name}' branch! The available variables are "
            f"{sorted(available_variables)}.",
            logger,
        )


def _create_mini_df_from_h5(
    file: h5py.File, group_name: str, variables: List[str]
) -> pd.DataFrame:
    """\
    [ Internal ] Load a "mini" `DataFrame` from a single HDF5 file.
    """
    _check_h5_group_has_variables(
        file=file, group_name=group_name, variables=variables
    )

    file_data: dict[str, npt.NDArray] = {}

    for full_variable_name in variables:
        logger.debug(f"Extracting variable '{full_variable_name!s}'...")

        variable_name = full_variable_name.split("/")[-1]

        h5_branch = file[f"{group_name}/{variable_name}"]

        if not isinstance(h5_branch, h5py.Dataset):
            _error(
                OscanaError,
                f"Variable '{group_name}/{variable_name}' is not a dataset!",
                logger,
            )

        file_data[variable_name] = np.asarray(h5_branch[()])

    return pd.DataFrame(file_data)


def _hdf5_file_loader(
    file_path: Path, variables: List[str]
) -> _FileLoaderResult:
    """\
    [ Internal ] Load data from a HDF5 file.
    """
    with h5py.File(file_path, "r") as h5_file:
        logger.debug(f"Opened {file_path!s} using `h5py`.")

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
    ) -> Optional[Exception]:
        """\
        [ Internal ] Update the parent `DataHandler` with the loaded data and 
        metadata.
        """
        if not len(mini_data_dfs):
            logger.warning(
                "No data was loaded from the provided files! The `DataHandler` "
                "will not be updated."
            )
            return None

        try:
            # Note: A file carrying cuts is rejected outright when there is no
            #       cuts table (see `_post_file_loader_checks`), so cuts never
            #       end up in the data table.
            new_cuts_table = self._parent._cuts_table

            if self._parent.has_cuts_table:
                new_cuts_table = pd.concat(
                    [self._parent._cuts_table, *mini_cuts_dfs],
                    ignore_index=True,
                )

            new_data_table = pd.concat(
                [self._parent._data_table, *mini_data_dfs], ignore_index=True
            )

        except Exception as output_error:
            self._cache.difference_update(cache)
            _warn(
                RuntimeWarning,
                f"An error occurred while updating the `DataHandler` "
                f"{ERROR_IN_WARN_FORMAT.format(error=output_error)}!",
                logger=logger,
            )
            return output_error

        else:
            # ~ only touch the parent once both tables are known to be valid
            self._parent._data_table = new_data_table
            self._parent._cuts_table = new_cuts_table
            self._parent._t_metadata = transform_metadata
            self._parent._f_metadata.extend(file_metadata)

        return None

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

        cache_proxy: Set[str] = set()  # ~ a proxy
        mini_data_dfs: List[pd.DataFrame] = []
        mini_cuts_dfs: List[pd.DataFrame] = []
        f_metadata_proxy: List[FileMetadata] = []
        t_metadata_proxy = self._parent._t_metadata  # ~ a proxy

        data_vars_proxy = self.get_vars_data_table().copy()
        cuts_vars_proxy = self.get_vars_cuts_table().copy()

        # Note: Variable naming here is not the best. `name` refers to the user
        #       -provided file name (in ".env" or a path), while `path` refers
        #       to the actual expanded file path.

        for name in user_files:
            name = str(name)  # ~ the name should be a string for the cache

            try:
                path = _resolve_file_directory(file=name)

                if str(path) in self._cache:
                    logger.warning(
                        f"File '{name}' has already been loaded! Skipping..."
                    )
                    continue

                logger.debug(f"Trying to load data from '{name}'...")
                result = file_loader_func(path, self._parent._variables)

                if not (
                    len(self._parent._f_metadata)
                    or len(mini_data_dfs)
                    or self.get_n_rows_data_table()
                ):
                    # Only update the transform metadata like this if there are
                    # no file loaded yet.
                    #
                    # Here we need to check the parent's file metadata because
                    # we only overwrite the transform metadata if there are no
                    # files loaded into the `DataHandler`!
                    t_metadata_proxy = result["transform_metadata"]

                _post_file_loader_checks(
                    file_name=name,
                    result=result,
                    data_vars_proxy=data_vars_proxy,
                    cuts_vars_proxy=cuts_vars_proxy,
                    current_t_metadata=t_metadata_proxy,
                    has_cuts_table=self._parent.has_cuts_table,
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
                self._cache.add(str(path))  # ~ this is important!
                cache_proxy.add(str(path))

                logger.info(
                    f"Loaded {len(result['mini_data_df'])} rows from "
                    f"'{name}'."
                )

                # ~ every frame goes in with the same column order, so `concat`
                #   never has to align them later on
                mini_data_dfs.append(
                    _reorder_columns(
                        df=result["mini_data_df"], columns=data_vars_proxy
                    )
                )
                mini_cuts_dfs.append(
                    _reorder_columns(
                        df=result["mini_cuts_df"], columns=cuts_vars_proxy
                    )
                )
                f_metadata_proxy.extend(result["file_metadata"])

        update_error = self._update_parent(
            cache=cache_proxy,
            mini_data_dfs=mini_data_dfs,
            mini_cuts_dfs=mini_cuts_dfs,
            transform_metadata=t_metadata_proxy,
            file_metadata=f_metadata_proxy,
        )

        if update_error is not None:
            exceptions_.append(update_error)
        elif len(mini_data_dfs):
            logger.info(
                f"Finished loading {len(mini_data_dfs)} file(s). The data "
                f"table now has {self.get_n_rows_data_table()} rows and "
                f"{self.get_n_vars_data_table()} variables."
            )

        if len(exceptions_):
            _warn(
                RuntimeWarning,
                f"Loading finished with {len(exceptions_)} errors. Some files "
                "may have failed to load. (See the above exceptions.)",
                logger=logger,
            )

    @override
    def _init_data_table(self) -> pd.DataFrame:
        """\
        [ Internal ] Initialise the data table.
        """
        return pd.DataFrame()

    @override
    def _init_cuts_table(self) -> pd.DataFrame:
        """\
        [ Internal ] Initialise the cuts table.
        """
        return pd.DataFrame()

    @override
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

    @override
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

    @override
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

    @override
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

        if out_file.suffix != ".h5":
            _error(
                OscanaError,
                f"The out file '{out_file!s}' is not an HDF5 file! Provide a "
                "path to a file with the '.h5' extension.",
                logger,
            )

        if ensure_parent_dir:
            out_file.parent.mkdir(parents=True, exist_ok=True)
        elif not out_file.parent.is_dir():
            _error(
                OscanaError,
                f"The parent directory '{out_file.parent!s}' does not exist! "
                "Set `ensure_parent_dir=True` to create it automatically.",
                logger,
            )

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

        logger.info(
            f"Wrote {self.get_n_rows_data_table()} rows and "
            f"{self.get_n_vars_data_table()} variables to '{out_file!s}'."
        )

    @override
    def get_n_rows_data_table(self) -> int:
        """\
        Get the length of the data table.
        """
        return len(self._parent._data_table)

    @override
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

    @override
    def get_n_vars_data_table(self) -> int:
        """\
        Get the number of variables in the data table.
        """
        return len(self._parent._data_table.columns)

    @override
    def get_n_vars_cuts_table(self) -> int:
        """\
        Get the number of variables in the cuts table.
        """
        if self._parent.has_cuts_table:
            return len(self._parent._cuts_table.columns)

        return 0

    @override
    def get_vars_data_table(self) -> List[str]:
        """\
        Get the list of variable names in the data table.
        """
        return list(self._parent._data_table.columns)

    @override
    def get_vars_cuts_table(self) -> List[str]:
        """\
        Get the list of variable names in the cuts table.
        """
        if self._parent.has_cuts_table:
            return list(self._parent._cuts_table.columns)

        return []
