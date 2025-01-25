"""\
oscana / data / sntp_loader.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import logging

import pandas as pd

from . import functions
from .metadata import FileMetadata
from ..utils import _error, OscanaError

# =============================== [ Logging  ] =============================== #

logger = logging.getLogger("Root")

# ============================= [ File Loader  ] ============================= #


class FileLoader:
    """\
    File Loader.

    Usage
    -----

    ```python
    from oscana import FileLoader

    # Create a new FileLoader object
    file = FileLoader(variables=["x", "y", "z"])

    # Load data from files
    file.load_from_files("file1.root", "file2.root")

    # Apply transformations
    file.transform(_some_cut, _another_cut)

    Notes
    -----

    Check which file loader method is currently being used: this information is
    stored in the `_file_loader` class attribute.
    """

    __slots__ = [
        "_variables",
        "_cut_variables",
        "_enable_cut_bools",
        "_data",
        "_cuts_table",
        "_cuts_bool",
        "_metadata",
        "_transforms",
    ]

    _sntp_file_loader: Callable[
        [list[str], str], tuple[pd.DataFrame, FileMetadata]
    ] = functions._v1_naive_loader

    _dst_file_loader: Callable[
        [list[str], str], tuple[pd.DataFrame, FileMetadata]
    ] = functions._v1_naive_loader

    def __init__(
        self, variables: list[str], cut_variables: list[str] | None = None
    ) -> None:
        """\
        Create a new SNTPFileLoader object.
        
        Parameters
        ----------
        variables : list[str]
            The list of variables to load from the files.
        """
        super().__init__()

        if cut_variables is None:
            cut_variables = []  # Default value

        if functions._check_for_repeats(variables):
            _error(
                OscanaError,
                f"Oscana.{self.__class__.__name__} got repeated variables.",
                logger,
            )

        if functions._check_for_repeats(cut_variables):
            raise OscanaError(
                f"Oscana.{self.__class__.__name__}: "
                "does not accept repeated cut variables."
            )

        self._variables: list[str] = variables
        self._cut_variables: list[str] = cut_variables

        self._enable_cut_bools: bool = True

        self._data: pd.DataFrame = pd.DataFrame()  # Placeholder
        self._cuts_table: pd.DataFrame = pd.DataFrame()  # Placeholder
        self._cuts_bool: pd.DataFrame = pd.DataFrame()  # Placeholder
        self._metadata: list[FileMetadata] = []

        self._transforms: list[str] = []

    def load_from_sntp(self, files: list[str]) -> None:
        """\
        Load data from SNTP files.
        
        Parameters
        ----------
        files : list[str]
            The list of files to load data from.
        """
        if functions._check_for_repeats(files):
            _error(
                OscanaError,
                f"Oscana.{self.__class__.__name__} got repeated files.",
                logger,
            )

        for file in files:
            data, metadata = FileLoader._sntp_file_loader(self._variables, file)

            # Note: This is a problem because we are getting the metadata twice.
            #       This needs to be fixed.
            # TODO: Fix this.
            cuts, _ = FileLoader._sntp_file_loader(self._cut_variables, file)

            cuts_bool = pd.DataFrame(
                True,
                columns=self._cut_variables,
                index=data.index,
                # Note: Here we use a sparse data type to save memory. In this
                #       version the fill value used is True (i.e. True values
                #       are ignored).
                dtype=pd.SparseDtype(bool, fill_value=True),
            )

            self._metadata.append(metadata)
            self._data = pd.concat([self._data, data]).reset_index(drop=True)
            self._cuts_table = pd.concat([self._cuts_table, cuts]).reset_index(
                drop=True
            )
            self._cuts_bool = pd.concat([self._cuts_bool, cuts_bool])

            logger.debug(f"Loaded data from '{file}'.")

    def transform(
        self, transforms: list[Callable[[pd.DataFrame], None]]
    ) -> None:
        """\
        Apply transformations to the data.
        
        Parameters
        ----------
        transforms : list[Callable[[pd.DataFrame], None]]
            The list of transformations to apply to the data.

        Notes
        -----
        "Transforms" include cuts, normalisations, etc. They are applied to the
        data in the order they are provided.
        """
        for transform in transforms:
            logger.debug(
                f"Applying '{transform.__name__}' to the data in {self!s}."
            )
            self._transforms.append(transform.__name__)
            transform(self._data)

        logger.info(
            f"{len(transforms)} transformations applied to the data in "
            f"{self!s}."
        )

    def copy(self) -> "FileLoader":
        _copy = FileLoader(
            variables=self._variables, cut_variables=self._cut_variables
        )
        raise NotImplementedError

    def print_metadata(self) -> None:
        """\
        Print the metadata of the loaded files.
        """
        for metadata in self._metadata:
            metadata.print()

    def is_using_cut_bools(self) -> bool:
        """\
        Check if the cut boolean table is enabled.
        
        Returns
        -------
        bool
            True if the cut boolean table is enabled, False otherwise.
        """
        return self._enable_cut_bools

    def get_applied_transforms(self) -> tuple[str, ...]:
        """\
        Gets all of the applied transformations.
        
        Returns
        -------
        tuple[str, ...]
            The tuple of the applied transformations.
        """
        return tuple(self._transforms)

    def get_sntp_loader_method(self) -> str:
        """\
        Gets the current file loader method.
        
        Returns
        -------
        str
            The current file loader method.
        """
        return self._sntp_file_loader.__name__

    def get_dst_loader_method(self) -> str:
        """\
        Gets the current file loader method.
        
        Returns
        -------
        str
            The current file loader method.
        """
        return self._dst_file_loader.__name__

    def toggle_cut_bools(self) -> None:
        """\
        Enable the cut boolean table.
        """
        self._enable_cut_bools = not self._enable_cut_bools

    def __str__(self) -> str:
        return (
            f"Oscana.{self.__class__.__name__}("
            f"n_files={len(getattr(self, "_metadata", []))}, "
            f"n_variables={len(getattr(self, "_variables", []))}, "
            f"n_transforms={len(getattr(self, "_transforms", []))}"
            ")"
        )

    def __repr__(self) -> str:
        return str(self)
