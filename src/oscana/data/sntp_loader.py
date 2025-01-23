"""\
oscana / data / sntp_loader.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from typing import Literal, Any, Callable

from datetime import datetime
from dataclasses import dataclass

from numpy import typing as npt
import uproot
import pandas as pd

from ..utils import OscanaError, _get_dir_from_env
from .base_loader import DataLoaderBase


# =============================== [ Metadata ] =============================== #


@dataclass(frozen=True)
class SNTPFileMetadata:
    """\
    [ Internal ] 
    
    Dataclass for storing metadata of SNTP files.
    
    Parameters
    ----------
    file_name : str
        The name of the file.

    experiment : Literal["MINOS", "NOvA", "DUNE"]
        The experiment the file is from.

    detector : Literal["Near", "Far"]
        The detector: either "near" or "far".

    file_type : Literal["MC", "Data"]
        The type of the file: either MC or (experiment) data.

    nu_source : Literal["Beam", "Atmospheric"]
        The source of the neutrinos: either "beam" or "atmpspheric".

    version : str
        Analysis or MC version e.g. "Dogwood 7".
    
    start_datetime : datetime
        The start date and time of the data collection.

    end_datetime : datetime
        The end date and time of the data collection.

    run_number : int
        The run number of the data collection.   
    """

    # Basic Information
    file_name: str
    experiment: Literal["MINOS"]  # Currently only MINOS is supported.
    detector: Literal["Near", "Far"]

    # Contents
    file_type: Literal["MC", "Data"]
    nu_source: Literal["Beam", "Atmospheric"]
    version: str

    # Date and Time
    start_datetime: datetime
    end_datetime: datetime

    # Run Information
    run_number: int

    def __str__(self) -> str:
        return f"Oscana.SNTPFileMetadata(file='{self.file_name}')"

    def __repr__(self) -> str:
        return str(self)


# ============================= [ File Loader  ] ============================= #


def _check_for_repeats(some_list: list[Any]) -> bool:
    """\
    [ Internal ]

    Check if the list has any repeated elements.
    """
    return len(some_list) != len(set(some_list))


def _get_file_metadata(file: str) -> SNTPFileMetadata:
    """\
    [ Internal ]

    Get the metadata of the file.
    """
    # Temporary metadata -- TODO: Implement a proper metadata loader.
    return SNTPFileMetadata(
        file_name=file,
        experiment="MINOS",
        detector="Far",
        file_type="MC",
        nu_source="Beam",
        version="Dogwood 7",
        start_datetime=datetime.now(),
        end_datetime=datetime.now(),
        run_number=0,
    )


# The functions below are all the direct SNTP file loaders.


def _v1_naive_loader(
    variables: list[str], file: str
) -> tuple[pd.DataFrame, SNTPFileMetadata]:
    """\
    [ Internal ]

    V1 File Loader. A naive loader - s  low, but works.
    """

    file_dir = _get_dir_from_env(file=file)

    uproot_file = uproot.open(file_dir)
    metadata = _get_file_metadata(file=file)

    data_dict: dict[str, npt.NDArray] = {}

    for variable in variables:
        base = variable.split("/")[0]
        key = "/".join(variable.split("/")[1:])

        # Note: I have added some `pyright` comments to suppress annoying
        #       warnings.

        data_dict[key] = uproot_file[base][
            key
        ].arrays(  # pyright: ignore[reportAttributeAccessIssue]
            library="np"
        )[
            key.split("/")[-1]
        ]

    uproot_file.close()  # pyright: ignore[reportAttributeAccessIssue]

    return pd.DataFrame(data_dict), metadata


class SNTPFileLoader(DataLoaderBase):
    """\
    SNTP File Loader.

    Usage
    -----

    ```python
    from oscana import SNTPFileLoader

    # Create a new SNTPFileLoader object
    file = SNTPFileLoader(variables=["x", "y", "z"])

    # Load data from files
    file.load_from_files("file1.root", "file2.root")

    # Apply transformations
    file.transform(_some_cut, _another_cut)

    Notes
    -----

    Check which file loader method is currently being used: this information is
    stored in the `_file_loader` class attribute.
    """

    __slots__ = ["_variables", "_data", "_metadata", "_transforms"]

    _file_loader: Callable[
        [list[str], str], tuple[pd.DataFrame, SNTPFileMetadata]
    ] = _v1_naive_loader

    def __init__(self, variables: list[str]) -> None:
        """\
        Create a new SNTPFileLoader object.
        
        Parameters
        ----------
        variables : list[str]
            The list of variables to load from the files.
        """
        super().__init__()

        if _check_for_repeats(variables):
            raise OscanaError(
                f"Oscana.{self.__class__.__name__}: "
                "does not accept repeated variables."
            )

        self._variables: list[str] = variables

        self._data: pd.DataFrame = pd.DataFrame()  # Placeholder
        self._metadata: list[SNTPFileMetadata] = []

        self._transforms: list[str] = []

    def load_from_files(self, files: list[str]) -> None:

        if _check_for_repeats(files):
            raise OscanaError(
                f"Oscana.{self.__class__.__name__}: "
                "does not accept repeated files."
            )

        for file in files:
            data, metadata = SNTPFileLoader._file_loader(self._variables, file)

            self._metadata.append(metadata)
            self._data = pd.concat([self._data, data]).reset_index(drop=True)

    def transform(
        self, transforms: list[Callable[[pd.DataFrame], None]]
    ) -> None:
        for transform in transforms:
            self._transforms.append(transform.__name__)
            transform(self._data)

    def copy(self) -> "SNTPFileLoader":
        _copy = SNTPFileLoader(variables=self._variables)
        raise NotImplementedError

    def get_applied_transforms(self) -> tuple[str, ...]:
        """\
        Gets all of the applied transformations.
        
        Returns
        -------
        tuple[str, ...]
            The tuple of the applied transformations.
        """
        return tuple(self._transforms)

    def get_file_loader_method(self) -> str:
        """\
        Gets the current file loader method.
        
        Returns
        -------
        str
            The current file loader method.
        """
        return self._file_loader.__name__
