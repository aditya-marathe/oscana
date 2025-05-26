"""\
oscana / data / data_handler.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = ["DataHandler"]

from typing import TypeVar, Generic

import logging

from ..logger import _error, _warn
from .io_base import DataIOStrategy
from .t_metadata import TransformMetadata
from .f_metadata import FileMetadata
from .transform import TransformBase
from ..utils import import_plugins, OscanaError

# =============================== [ Logging  ] =============================== #

_logger = logging.getLogger("Root")

# ============================= [ Load Plugins ] ============================= #

# Note: Not sure this is the best way to use the Plugin Architecture in Python.
#       It seems to be working though...

plugins = import_plugins(file=__file__)

# ============================= [ Data Handler ] ============================= #


T = TypeVar("T")


class DataHandler(Generic[T]):
    """\
    Data Handler
    ------------

    """

    __slots__ = [
        "_data_io",
        "_variables",
        "_has_cuts_table",
        "_t_metadata",
        "_f_metadata",
        "_data_table",
        "_cuts_table",
    ]

    def __init__(
        self,
        variables: list[str],
        data_io: str = "PandasIO",
        make_cut_bool_table: bool = False,
    ) -> None:
        """\
        Initialise a `DataHandler` object.

        Parameters
        ----------
        variables : list[str]
            List of variables.
        
        data_io : type[DataIOStrategyABC]
            Data IO strategy.

        make_cut_bool_table : bool, optional
            Whether to make a cuts table, by default False.
        """
        # (1) Get the Data IO plugin.

        data_io_plugin: type[DataIOStrategy[T]] | None = plugins.get(
            data_io, None
        )

        if data_io_plugin is None:
            _error(
                OscanaError,
                (f"Data IO strategy '{data_io}' not found in the plugins."),
                _logger,
            )

        # (2) Initialise the instance variables.

        self._data_io: DataIOStrategy[T] = data_io_plugin(parent=self)

        self._variables = variables
        self._has_cuts_table = bool(make_cut_bool_table)  # Just to be sure.

        self._t_metadata = TransformMetadata()
        self._f_metadata: list[FileMetadata] = []

        self._data_table: T = self.io._init_data_table()
        self._cuts_table: T = self.io._init_cuts_table()

    def apply_transforms(self, transforms: list[TransformBase]) -> None:
        """\
        Apply the transforms to the data.
        
        Parameters
        ----------
        transforms : list[TransformBase]
            List of transforms to apply.
        """
        n_errors: int = 0

        for i, transform in enumerate(transforms):
            len_before = self.io.get_data_length()

            try:
                self._data_table, self._cuts_table = transform(dh=self)
            except Exception as e:
                n_errors += 1

                _warn(
                    RuntimeWarning,
                    f"Error while applying transform `{transform}`. "
                    f"{e.__class__.__name__}: {e}",
                    _logger,
                )

            else:
                self._t_metadata._add_transform(transform=transform)

            _logger.info(
                f"({i + 1}/{len(transforms)}) Applied the transform "
                f"`{transform}` to the data with {n_errors} errors. "
                f"Number of Rows {len_before} -> {self.io.get_data_length()}."
            )

        if n_errors > 0:
            _error(
                OscanaError,
                f"Failed to apply {n_errors} transforms to the data. (See above"
                " warnings.)",
                _logger,
            )

    def print_handler_info(self) -> None:
        """\
        Print handler information.
        """
        info = self.io._get_strategy_info()

        unknown: str = "???"

        print("Data IO\n" + "-" * 7)
        print("\t- IO Strategy Class : " + str(self.io))
        print("\t- SNTP Loader       : " + info.get("SNTP Loader", unknown))
        print("\t- uDST Loader       : " + info.get("uDST Loader", unknown))
        print("\t- HDF5 Loader       : " + info.get("HDF5 Loader", unknown))
        print("\t- HDF5 Writer       : " + info.get("HDF5 Writer", unknown))
        print("\nSettings\n" + "-" * 8)
        print(
            "\t- Cuts Table : "
            + ("Enabled" if self._has_cuts_table else "Disabled")
        )

    def print_metadata(self) -> None:
        """\
        Print metadata.
        """
        self._t_metadata.print()
        print()
        for fm in self._f_metadata:
            fm.print()

        print()  # Add a newline for better readability.

    @staticmethod
    def print_available_plugins() -> None:
        """\
        Print available plugins.
        """
        print("Available Data IO Plugins")
        print("-------------------------")
        for plugin_name in plugins.keys():
            print(f"\t- '{plugin_name}'")

        if not plugins:
            print("\t[ No plugins ]")

    @property
    def data(self) -> T:
        return self._data_table

    @property
    def io(self) -> DataIOStrategy[T]:
        return self._data_io

    @property
    def has_cuts_table(self) -> bool:
        return self._has_cuts_table

    def __str__(self) -> str:
        return (
            f"oscana.{self.__class__.__name__}("
            f"n_variables={len(self._variables)}, "
            f"n_transforms={len(self._t_metadata.transforms)}, "
            f"n_files={len(self._f_metadata)})"
        )

    def __repr__(self) -> str:
        return str(self)
