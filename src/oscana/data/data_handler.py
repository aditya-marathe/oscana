"""\
oscana / data / data_handler.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations


from typing import Any, List, Dict, NoReturn
from typing import Optional, Union
from typing import TypeVar, Generic

__all__ = ["DataHandler"]

import logging

from ..logger import _error, _warn
from .io_base import _DataIOStrategy
from .t_metadata import TransformMetadata
from .f_metadata import FileMetadata
from .transform import TransformBase
from .callback_base import DataCallbackBase
from ..utils import import_plugins, OscanaError
from ..escape import Style

# =============================== [ Logging  ] =============================== #

_logger = logging.getLogger("Root")

# ============================= [ Load Plugins ] ============================= #

plugins = import_plugins(file=__file__)

# =========================== [ Helper Functions ] =========================== #


def _check_has_cuts_table(has_cuts_table: bool) -> None:
    """\
    [ Internal ] Check if cuts table is enabled.
    
    Parameters
    ----------
    has_cuts_table : bool
        Whether cuts table is enabled.
    """
    if not has_cuts_table:
        _error(
            OscanaError,
            "Cuts table is not enabled for this `DataHandler` object!",
            _logger,
        )


def _check_data_locked(is_data_locked: bool) -> None:
    """\
    [ Internal ] Check if data is locked.
    
    Parameters
    ----------
    is_data_locked : bool
        Whether data is locked.
    """
    if is_data_locked:
        _error(
            OscanaError,
            "Data is locked, thus cannot be modified!",
            _logger,
        )


def _run_callbacks(
    dh: DataHandler,
    callbacks: list[DataCallbackBase],
    transform: TransformBase,
    is_before_transform: bool,
) -> int:
    """\
    [ Internal ] Run the registered callbacks and count errors.

    Parameters
    ----------
    dh : DataHandler
        The `DataHandler` instance.
    
    callbacks : list[DataCallbackBase]
        The list of callbacks to run.

    transform : TransformBase
        The transform that is being applied.

    is_before_transform : bool
        Whether the callbacks are for the before or after the transform.

    Returns
    -------
    int
        The number of errors encountered while running the callbacks.
    """
    n_errors = 0

    for callback in callbacks:
        try:
            if is_before_transform:
                callback.before_transform(dh=dh, transform=transform)
            else:
                callback.after_transform(dh=dh, transform=transform)
        except Exception as e:
            n_errors += 1

            stage = "before" if is_before_transform else "after"

            _warn(
                RuntimeWarning,
                f"Error while running callback `{callback}` "
                f"{stage} applying transform `{transform}`. "
                f"{e.__class__.__name__}: {e}",
                _logger,
            )

            continue

    return n_errors


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
        "_is_data_locked",
    ]

    def __init__(
        self,
        variables: List[str],
        data_io: str = "PandasIO",
        make_cut_bool_table: bool = False,
    ) -> None:
        """\
        Initialise a `DataHandler` object.

        Parameters
        ----------
        variables : List[str]
            List of variables.
        
        data_io : type[DataIOStrategyABC]
            Data IO strategy.

        make_cut_bool_table : bool, optional
            Whether to make a cuts table, by default False.
        """
        # (1) Get the Data IO plugin.

        data_io_plugin: type[_DataIOStrategy[T]] | None = plugins.get(
            data_io, None
        )

        if data_io_plugin is None:
            _error(
                OscanaError,
                (f"Data IO strategy '{data_io}' not found in the plugins."),
                _logger,
            )

        # (2) Initialise the instance variables.

        self._data_io: _DataIOStrategy[T] = data_io_plugin(parent=self)

        if not len(variables):
            _error(
                OscanaError,
                "The list of variables is empty! Specify at least one variable "
                "to load.",
                _logger,
            )

        self._variables = list(dict.fromkeys(variables))  # ~ remove duplicates
        if len(self._variables) != len(variables):
            _warn(
                RuntimeWarning,
                "Duplicate variables found in the input list! Duplicates have "
                "been removed.",
                _logger,
            )

        self._has_cuts_table = bool(make_cut_bool_table)

        self._t_metadata = TransformMetadata()
        self._f_metadata: list[FileMetadata] = []

        self._data_table: T = self.io._init_data_table()
        self._cuts_table: T = self.io._init_cuts_table()

        self._is_data_locked: bool = True

    def lock_data(self) -> None:
        """\
        Lock the data, preventing any modifications.
        """
        self._is_data_locked = True

        _logger.info("Data has been locked!")

    def unlock_data(self) -> None:
        """\
        Unlock the data, allowing modifications.

        Note
        ----
        Realistically, this should never have to be used by the user.
        """
        self._is_data_locked = False

        _warn(
            RuntimeWarning, "Data has been unlocked for modification!", _logger
        )

    def is_data_locked(self) -> bool:
        """\
        Check if the data is locked.

        Returns
        -------
        bool
            Whether the data is locked.
        """
        return self._is_data_locked

    def apply_transforms(
        self,
        transforms: List[TransformBase],
        callbacks: Optional[List[DataCallbackBase]] = None,
    ) -> None:
        """\
        Apply the transforms to the data.
        
        Parameters
        ----------
        transforms : List[TransformBase]
            List of transforms to apply.

        callbacks : Optional[List[DataCallbackBase]] | None
            List of callbacks to call after each transform. Defaults to `None`.
        """
        # (1) Input Validation.
        if callbacks is None:
            callbacks = []

        n_tf_errors: int = 0
        n_cb_errors: int = 0

        for i, transform in enumerate(transforms):
            len_before = self.io.get_n_rows_data_table()

            # (2) Run callbacks before the transform.
            n_cb_errors += _run_callbacks(
                dh=self,
                callbacks=callbacks,
                transform=transform,
                is_before_transform=True,
            )

            # (3) Unlock, Transform, Lock.
            try:
                self._data_table, self._cuts_table = transform(dh=self)
            except Exception as e:
                n_tf_errors += 1

                _warn(
                    RuntimeWarning,
                    f"Error while applying transform `{transform}`. "
                    f"{e.__class__.__name__}: {e}",
                    _logger,
                )

                continue

            # (4) Save transform metadata.
            self._t_metadata._add_transform(transform=transform)

            # (5) Run callbacks after the transform.
            n_cb_errors += _run_callbacks(
                dh=self,
                callbacks=callbacks,
                transform=transform,
                is_before_transform=False,
            )

            _logger.info(
                f"({i + 1}/{len(transforms)}) Applied the transform "
                f"`{transform}` to the data with {n_tf_errors} errors. "
                f"Number of Rows {len_before} -> "
                f"{self.io.get_n_rows_data_table()}."
            )

        # Note: We do not want to interrupt the loop due to errors!

        if (n_tf_errors > 0) or (n_cb_errors > 0):
            _error(
                OscanaError,
                f"Failed to apply {n_tf_errors}/{len(transforms)} transforms "
                "to the data, or failed to run callbacks! "
                "(Check above warnings.)",
                _logger,
            )

    def get_transforms_dict(self) -> Dict[str, Dict[str, Any]]:
        """\
        Get the transforms as a dictionary.
        
        Returns
        -------
        dict[str, dict[str, Any]]
            A dictionary containing the metadata. The keys are the name of the 
            transform and the values are keyword arguments passed to the 
            function.
        """
        return self._t_metadata.to_dict()

    def print_metadata(self) -> None:
        """\
        Print metadata.
        """
        self._t_metadata.print()

        print()

        for fm in self._f_metadata:
            fm.print()

        print()

    @staticmethod
    def print_available_plugins() -> None:
        """\
        Print available plugins.
        """
        print(Style.BD + "Available Data IO Plugins" + Style.R)
        print(Style.BD + "-------------------------" + Style.R)
        for plugin_name in plugins.keys():
            print(f"\t- '{plugin_name}'")

        if not plugins:
            print(f"\t{Style.FG[8]}[ No plugins ]{Style.R}")

    @property
    def data(self) -> T:
        return self._data_table

    @data.setter
    def data(self, value: T) -> None:
        _check_data_locked(is_data_locked=self._is_data_locked)
        self._data_table = value

    @property
    def cuts(self) -> Union[T, NoReturn]:
        _check_has_cuts_table(has_cuts_table=self._has_cuts_table)
        return self._cuts_table

    @cuts.setter
    def cuts(self, value: T) -> None:
        _check_data_locked(is_data_locked=self._is_data_locked)
        _check_has_cuts_table(has_cuts_table=self._has_cuts_table)
        self._cuts_table = value

    @property
    def io(self) -> _DataIOStrategy[T]:
        return self._data_io

    @property
    def has_cuts_table(self) -> bool:
        return self._has_cuts_table

    def __str__(self) -> str:
        return (
            f"oscana.{self.__class__.__name__}("
            f"n_variables={self.io.get_n_vars()}, "
            f"n_transforms={len(self._t_metadata.transforms)}, "
            f"n_files={len(self._f_metadata)})"
        )

    def __repr__(self) -> str:
        return str(self)
