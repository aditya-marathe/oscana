"""\
oscana / data / data_handler.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = ["DataHandler"]

from typing import TypeVar, Any

import logging

from .io_base import DataIOStrategy
from .t_metadata import TransformMetadata
from .f_metadata import FileMetadata
from ..utils import import_plugins, OscanaError, _error

# ================================ [ Logger ] ================================ #

logger = logging.getLogger("Root")

# ============================= [ Load Plugins ] ============================= #

# Note: Not sure this is the best way to use the Plugin Architecture in Python.
#       It seems to be working though...

plugins = import_plugins(file=__file__)

# ============================= [ Data Handler ] ============================= #


T = TypeVar("T", bound=DataIOStrategy)


class DataHandler:
    def __init__(
        self,
        variables: list[str],
        data_io: str = "PandasIO",
        make_cut_bool_table: bool = False,
    ) -> None:
        """\
        Data Handler.

        Parameters
        ----------
        variables : list[str]
            List of variables.
        
        data_io : type[DataIOStrategyABC]
            Data IO strategy.

        make_cut_bool_table : bool, optional
            Whether to make a cuts table, by default False.
        """
        # Get the Data IO plugin.
        data_io_plugin: type[DataIOStrategy] = plugins.get(data_io, None)
        if data_io_plugin is None:
            _error(
                OscanaError,
                (f"Data IO strategy '{data_io}' not found in the plugins."),
                logger,
            )
        self._data_io: DataIOStrategy = data_io_plugin(parent=self)

        self._variables = variables
        self._has_cuts_table = bool(make_cut_bool_table)  # Just to be sure.

        self._t_metadata = TransformMetadata()
        self._f_metadata: list[FileMetadata] = []

        self._data_table = self.io._init_data_table()
        self._cuts_table = self.io._init_cuts_table()

    def print_handler_info(self) -> None:
        """\
        Print handler information.
        """
        info = self.io._get_strategy_info()

        print("Data IO\n" + "-" * 7)
        print("\t- IO Strategy Class : " + str(self.io))
        print("\t- SNTP Loader       : " + info.get("SNTP Loader", "???"))
        print("\t- uDST Loader       : " + info.get("uDST Loader", "???"))
        print("\t- HDF5 Loader       : " + info.get("HDF5 Loader", "???"))
        print("\t- HDF5 Writer       : " + info.get("HDF5 Writer", "???"))
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
    def io(self) -> DataIOStrategy:
        return self._data_io

    @property
    def has_cuts_table(self) -> bool:
        return self._has_cuts_table

    def __str__(self) -> str:
        return (
            f"Oscana.{self.__class__.__name__}("
            f"n_variables={len(self._variables)}, "
            f"n_transforms={len(self._t_metadata.transforms)}, "
            f"n_cuts={len(self._t_metadata.cuts)}, "
            f"n_files={len(self._f_metadata)})"
        )

    def __repr__(self) -> str:
        return str(self)
