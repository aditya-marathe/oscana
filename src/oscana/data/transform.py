"""\
oscana / data / transform.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = ["TransformBase"]

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data_handler import DataHandler

# ============================== [ Transforms ] ============================== #


class TransformBase(ABC):
    """\
    Transform Base Class
    --------------------
    
    This is the base class for all transform functions. It is used to register
    transform functions in the `DataHandler` class.
    """

    def __init__(self, **kwargs: Any) -> None:
        """\
        Initialize the transform function.

        Parameters
        ----------
        kwargs: dict[str, Any]
            The keyword arguments to pass to the transform function.
        """
        self._kwargs: dict[str, Any] = kwargs

    @abstractmethod
    def _transform(self, dh: DataHandler) -> tuple[Any, Any]:
        """\
        Transform the data.

        Parameters
        ----------
        dh: DataHandler
            The data handler object.

        Returns
        -------
        tuple[Any, Any]
            The data and cuts tables.
        """
        pass

    def __call__(self, dh: DataHandler) -> tuple[Any, Any]:
        """\
        Call the transform function.

        Parameters
        ----------
        dh: DataHandler
            The data to transform.

        Returns
        -------
        tuple[Any, Any]
            The data and cuts tables.
        """
        return self._transform(dh=dh)

    def __str__(self) -> str:
        """\
        String representation of the transform function.

        Returns
        -------
        str
            The string representation of the transform function.
        """
        return (
            f"oscana.{self.__class__.__name__}("
            + ", ".join([f"{k}={repr(v)}" for k, v in self._kwargs.items()])
            + ")"
        )

    def __repr__(self) -> str:
        return str(self)
