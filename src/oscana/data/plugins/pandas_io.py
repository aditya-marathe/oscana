"""\
oscana / data / plugins / pandas_v1.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

The Pandas-based input/output strategy.

"""

from __future__ import annotations

from typing import Tuple, List, Literal, Callable
from typing import Any, Union, Optional
from typing import TYPE_CHECKING, TypeAlias

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
from ..io_base import _DataIOStrategy, _SupportedCompressionType
from ..t_metadata import TransformMetadata
from ..f_metadata import FileMetadata
from ...utils import (
    OscanaError,
    get_func_lookup,
    _get_dir_from_env,
)

if TYPE_CHECKING:
    from ..data_handler import DataHandler


# =============================== [ Logging  ] =============================== #

logger = logging.getLogger("Root")


# ============================== [ Constants  ] ============================== #

SAVE_FILE_FORMAT = "{timestamp}_{name}.{format}"

ERROR_IN_WARN_FORMAT = "[E//{error}]"  # ~ so I can Ctrl+F for it :)


_FileLoaderResult: TypeAlias = Tuple[
    pd.DataFrame, FileMetadata, TransformMetadata
]
_FileLoaderFuncType: TypeAlias = Callable[[Path, List[str]], _FileLoaderResult]

# =============================== [ Helpers  ] =============================== #


def _v1_naive_loader(
    variables: list[str], file: str
) -> _LoadedDataType[pd.DataFrame]:
    """\
    A simple way to load variables from a ROOT file using Uproot.
    
    Parameters
    ----------
    variables : list[str]
        List of variables to load from the ROOT file.
    
    file : str
        The name of the ROOT file to load the variables from.
        
    Returns
    -------
    LoadedDataType[pd.DataFrame]
        A tuple containing:
        - A `DataFrame` with the loaded variables.
        - A list of `FileMetadata` objects with the metadata of the files.
        - A `TransformMetadata` object with the metadata of the transforms.
    """
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

        try:
            base_branch = uproot_file[base]
        except uproot.KeyInFileError:
            _error(
                OscanaError,
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
                OscanaError,
                f"Variable '{key}' not found in '{file}'!",
                logger,
            )

    uproot_file.close()  # pyright: ignore[reportAttributeAccessIssue]

    logger.info(f"Extracted variables from '{file}'.")

    return pd.DataFrame(data_dict), [metadata], TransformMetadata()


def _v1_naive_loader_h5(
    variables: list[str], file: str | Path
) -> _LoadedDataType[pd.DataFrame]:
    """\
    A simple way to load variables from an HDF5 file using h5py.
    
    Parameters
    ----------
    variables : list[str]
        List of variables to load from the HDF5 file.

    file : str | Path
        The name or path of the HDF5 file to load the variables from.

    Returns
    -------
    LoadedDataType[pd.DataFrame]
        A tuple containing:
        - A `DataFrame` with the loaded variables.
        - A list of `FileMetadata` objects with the metadata of the files.
        - A `TransformMetadata` object with the metadata of the transforms.
    """
    logger.debug(
        f"Loading variables from '{file}' using the V1 Naive Loader (HDF5)."
    )

    variables_set: set[str] = {var.split("/")[-1] for var in variables}

    with h5py.File(file, "r") as h5_file:
        logger.info(f"Opened '{file}' using h5py.")

        # (1) Extract the metadata.

        t_branch = h5_file["metadata/transforms"]

        if not isinstance(t_branch, h5py.Dataset):
            _error(
                OscanaError,
                f"The 'transforms' branch in '{file}' is not a dataset!",
                logger,
            )

        t_branch_dict = json.loads(t_branch[()].decode("utf-8"))
        t_metadata = TransformMetadata.from_dict(meta_dict=t_branch_dict)

        f_branch = h5_file["metadata/files"]

        if not isinstance(f_branch, h5py.Dataset):
            _error(
                OscanaError,
                f"The 'files' branch in '{file}' is not a dataset!",
                logger,
            )

        f_branch_dict = json.loads(f_branch[()].decode("utf-8"))
        f_metadata = [
            FileMetadata.from_dict(meta_dict=f) for f in f_branch_dict
        ]

        del t_branch, f_branch, f_branch_dict, t_branch_dict

        # (2) Extract the data to a `DataFrame`.

        data_dict: dict[str, npt.NDArray] = {}

        data_branch = h5_file["data"]

        if not isinstance(data_branch, h5py.Group):
            _error(
                OscanaError,
                f"The 'data' branch in '{file}' is not a group!",
                logger,
            )

        for column_name in data_branch.keys():
            column_name = str(column_name)

            # Note: We should always be loading in all the "ana." variables, at
            #       least for this naive loader. Ideally, we should have some
            #       method of specifying which "ana." variables to load to avoid
            #       hogging up memory with useless data.

            is_not_ana_variable = not column_name.startswith("ana.")

            if is_not_ana_variable and (column_name not in variables_set):
                continue

            logger.debug(
                f"Extracting variable '{column_name}' from '{file}'..."
            )

            column = data_branch[column_name]

            if not isinstance(column, h5py.Dataset):
                _error(
                    OscanaError,
                    f"Variable 'data/{column_name}' in '{file}' is not a "
                    "dataset!",
                    logger,
                )

            data_dict[column_name] = column[:]

        cuts_branch = h5_file["cuts"]

        if not isinstance(cuts_branch, h5py.Group):
            _error(
                OscanaError,
                f"The 'cuts' branch in '{file}' is not a group!",
                logger,
            )

        for column_name in cuts_branch.keys():
            column_name = str(column_name)

            logger.debug(
                f"Extracting cut variable '{column_name}' from '{file}'..."
            )

            column = cuts_branch[column_name]

            if not isinstance(column, h5py.Dataset):
                _error(
                    OscanaError,
                    f"Variable 'cuts/{column_name}' in '{file}' is not a "
                    "dataset!",
                    logger,
                )

            data_dict[column_name] = column[:]

        return pd.DataFrame(data=data_dict), f_metadata, t_metadata


def _is_jagged_array(data: pd.Series) -> bool:
    """\
    Check if the data is a jagged array (i.e., a list of lists).

    Parameters
    ----------
    data : pd.Series
        The data to check.

    Returns
    -------
    bool
        True if the data is a jagged array, False otherwise.
    """
    # Note: This is a really crappy way to check for jagged arrays!
    # TODO: Make this less crappy.
    return isinstance(data.iloc[0], np.ndarray) and data.dtype == "object"


def _loader_function(
    helper_func: Callable[
        [list[str], str | Path], _LoadedDataType[pd.DataFrame]
    ],
    variables: list[str],
    files: list[str | Path],
) -> _LoadedDataType[pd.DataFrame]:
    """\
    [ Internal ] Generic loader function to handle loading data from files.

    Parameters
    ----------
    helper_func : Callable
        The helper function to use for loading the data.

    variables : list[str]
        List of variables to load from the files.

    files : list[str | Path]
        List of files to load the data from.
    
    Returns
    -------
    LoadedDataType[pd.DataFrame]
        A tuple containing:
        - A `DataFrame` with the loaded variables.
        - A list of `FileMetadata` objects with the metadata of the files.
        - A `TransformMetadata` object with the metadata of the transforms.
    """
    exceptions_ = []
    data_list: list[pd.DataFrame] = []
    f_metadata_list: list[FileMetadata] = []
    t_metadata_comp: TransformMetadata | None = None

    for file in files:
        try:
            data, f_metadata, t_metadata = helper_func(variables, file)

            if len(f_metadata_list) and all(
                fm != f_metadata_list[-1] for fm in f_metadata
            ):
                _error(
                    OscanaError,
                    "All files must have the same metadata! The metadata for "
                    f"{file} is different from the previous files.",
                    logger,
                )

            if len(f_metadata_list) and t_metadata_comp != t_metadata:
                _error(
                    OscanaError,
                    "All files must have the same transforms applied! The "
                    f"transforms for {file} are different from the previous "
                    "files.",
                    logger,
                )

            t_metadata_comp = t_metadata_comp or t_metadata
            data_list.append(data)
            f_metadata_list.extend(f_metadata)

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

    assert (
        t_metadata_comp is not None
    ), "Unreachable: Transform metadata is `None`! "

    return (pd.concat(data_list), f_metadata_list, t_metadata_comp)


# =========================== [ Dynamic Helpers  ] =========================== #


def hlp_20250205_to_hdf5(
    data: pd.DataFrame,
    cuts: pd.DataFrame | None,
    file_metadata: list[FileMetadata],
    transform_metadata: TransformMetadata,
    file_path: str | Path,
    compression: Literal["gzip", "lzf"] | None,
) -> None:
    """\
    [ Internal ]

    Name: HDF5 Writer V1
    """
    # (1) Check the file path.
    file_path = Path(file_path)

    if not (file_path.is_file and (file_path.suffix == ".h5")):
        _error(
            OscanaError,
            f"File '{file_path}' is not an HDF5 file! "
            + "Please provide a file with the '.h5' extension.",
            logger,
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # (2) Compression.
    compression_kwargs: dict[str, Any] = {}

    if compression is not None:
        compression_kwargs = {"compression": compression}

    with h5py.File(file_path, "w") as my_file:
        data_branch = my_file.create_group(name="data")

        for column_name in data.columns:
            data_type = data[column_name].dtype

            if _is_jagged_array(data=data[column_name]):
                data_type = h5py.special_dtype(
                    vlen=data[column_name].iloc[0].dtype
                )

            data_branch.create_dataset(
                name=column_name,
                dtype=data_type,
                data=data[column_name].to_numpy(),
                **compression_kwargs,
            )

        if (cuts is not None) and (not cuts.empty):
            cuts_branch = my_file.create_group(name="cuts")

            for column_name in cuts.columns:
                cuts_branch.create_dataset(
                    name=column_name,
                    dtype=cuts[column_name].dtype,
                    data=cuts[column_name].to_numpy(),
                    **compression_kwargs,
                )

        metadata_branch = my_file.create_group(name="metadata")

        metadata_branch.create_dataset(
            name="transforms",
            dtype=h5py.string_dtype(encoding="utf-8"),
            data=json.dumps(transform_metadata.to_dict()).encode("utf-8"),
        )

        metadata_branch.create_dataset(
            name="files",
            dtype=h5py.string_dtype(encoding="utf-8"),
            data=json.dumps([fm.to_dict() for fm in file_metadata]).encode(
                "utf-8"
            ),
        )


def _resolve_file_directory(file: Union[str, Path]) -> Path:
    """\
    [ Internal ] Get the file directory from enviornment variables or from
    a path.
    """
    if isinstance(file, str):
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

    mini_df = _create_mini_df_from_uproot(
        uproot_file=uproot_file,  # pyright: ignore[reportArgumentType]
        variables=variables,
    )
    uproot_file.close()  # pyright: ignore[reportAttributeAccessIssue]

    return mini_df, file_metadata, TransformMetadata()


def _create_mini_df_from_h5(
    h5_file: h5py.File, variables: List[str]
) -> pd.DataFrame:
    """\
    [ Internal ] Load a "mini" `DataFrame` from a single HDF5 file.
    """
    file_data: dict[str, npt.NDArray] = {}

    for full_variable_name in variables:
        pass

    return pd.DataFrame(file_data)


def _hdf5_file_loader(
    file_path: Path, variables: List[str]
) -> _FileLoaderResult:
    """\
    [ Internal ] Load data from a HDF5 file.
    """
    h5_file = h5py.File(file_path, "r")
    logger.debug(f"Opened {file_path!s} using `h5py`.")

    return NotImplemented()

    # TODO: Implement this.
    file_metadata = FileMetadata.from_dict(meta_dict=...)
    transform_metadata = TransformMetadata.from_dict(meta_dict=...)

    mini_df = _create_mini_df_from_h5(h5_file=h5_file, variables=variables)
    h5_file.close()

    return mini_df, file_metadata, transform_metadata


# ============================= [ IO Strategy  ] ============================= #


class PandasIO(_DataIOStrategy[pd.DataFrame]):
    def __init__(self, parent: "DataHandler[pd.DataFrame]") -> None:
        """\
        Pandas IO Strategy.
        """
        super().__init__(parent=parent)

    # Helpers

    def _load_from_files(
        self,
        user_files: List[Union[str, Path]],
        file_loader_func: _FileLoaderFuncType,
    ) -> None:
        """\
        [ Internal ] Load data from a list of files using the given file loader
        function.
        """
        exceptions_ = []
        prev_t_metadata: Optional[TransformMetadata] = None

        # Note: Variable naming here is not the best. `name` refers to the user
        #       -provided file name (in ".env" or a path), while `path` refers
        #       to the actual expanded file path.

        for name in user_files:
            if name in self._cache:
                logger.warning(
                    f"File '{name!s}' has already been loaded! Skipping..."
                )
                continue

            try:
                path = _resolve_file_directory(file=name)

                logger.debug(f"Trying to load data from '{name!s}'...")
                mini_df, f_metadata, t_metadata = file_loader_func(
                    path, self._parent._variables
                )

                # Check transform metadata...
                if prev_t_metadata is None:
                    prev_t_metadata = t_metadata
                elif prev_t_metadata != t_metadata:
                    _error(
                        OscanaError,
                        "All files must have the same transforms applied! The "
                        f"transforms for '{name!s}' are different from the "
                        "previous files.",
                        logger,
                    )

                # Update the data table and file metadata...
                self._parent._data_table = pd.concat(
                    [self._parent._data_table, mini_df],
                    ignore_index=True,
                )
                self._parent._f_metadata.append(f_metadata)

            except Exception as e:
                # ~ this ensures that the program does not stop if one of the
                #   files fail to load - more useful in interactive sessions
                _warn(
                    RuntimeWarning,
                    f"An error occurred while loading data from {name!s} "
                    f"{ERROR_IN_WARN_FORMAT.format(error=e)}!",
                    logger=logger,
                )
                exceptions_.append(e)
            else:
                self._cache.add(str(name))  # ~ this is important!

        if prev_t_metadata is not None:
            self._parent._t_metadata = prev_t_metadata  # ~ more book keeping

        if len(exceptions_):
            _warn(
                RuntimeWarning,
                f"Code finished with {len(exceptions_)} errors. Some files may "
                "have failed to load. (See the above exceptions.)",
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
        return NotImplemented()

    def from_hdf5(self, files: List[Union[str, Path]]) -> None:
        """\
        Load data from HDF5 files.
        
        Parameters
        ----------
        files : List[Union[str, Path]]
            List of files.
        """
        file_paths = _get_file_directories(files=files)
        self._load_from_files(
            user_files=files,
            file_paths=file_paths,
            file_loader_func=_hdf5_file_loader,
        )

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
