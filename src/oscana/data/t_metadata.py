"""\
oscana / data / t_metadata.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

This module contains the `TransformMetadata` class, which is used to keep track
of the cuts and transforms applied to the loaded data. This is useful when we
want to merge two datasets, and we want to ensure that the same cuts and
transforms have been applied to both datasets.
"""

from __future__ import annotations

# Note: There is a compatibility issue with `dataclasses`--in older versions of
#       Python, the `slots` parameter is not available. I will solve this issue
#       if it arises.

__all__ = []

from typing import Any

import logging
from dataclasses import dataclass, field

from ..logger import _error
from ..utils import OscanaError
from .transform import TransformBase

# =============================== [ Logging  ] =============================== #

_logger = logging.getLogger("Root")

# ============================ [ File Metadata  ] ============================ #


@dataclass(frozen=True, slots=True)
class TransformMetadata:
    """\
    Transform Metadata.

    Parameters
    ----------
    cuts : list[str]
        List of cuts.

    transforms : list[str]
        List of transforms.
    """

    transforms: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def _add_transform(self, transform: TransformBase) -> None:
        """\
        Add a new transform to the metadata.

        Parameters
        ----------
        transform: TransformBase
            The transform to add.

        Notes
        -----
        The function names must follow the Oscana naming convention:
            {`cut` / `tfm`}_{YYYYMMDD}_{name}
        """
        # Note: Here I am not checking if the transform has already been added!
        #       This will be taken care of by the `DataHandler` class.

        if not isinstance(transform, TransformBase):
            _error(
                TypeError,
                "Transform must be an instance of `TransformBase`.",
                _logger,
            )

        self.transforms.append(
            (
                transform.__class__.__name__,
                transform._kwargs,
            )
        )

    def _extract_transform_name(self, name: str) -> str:
        """\
        [ Internal ] Extract the transform name from the function name.
        
        Parameters
        ----------
        name : str
            The function name.

        Returns
        -------
        str
            The transform name.
        """
        return "_".join(name.split("_")[2:]).lower()

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """\
        Convert the metadata to a dictionary.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            A dictionary containing the metadata.
        """
        return {
            "transforms": [
                {
                    "name": data[0],
                    "shortened_name": self._extract_transform_name(data[0]),
                    "kwargs": data[1],
                }
                for data in self.transforms
            ]
        }

    @staticmethod
    def from_dict(data: dict[str, list[dict[str, Any]]]) -> TransformMetadata:
        """\
        Load the metadata from a dictionary.

        Parameters
        ----------
        data : dict[str, list[dict[str, Any]]]
            The dictionary containing the metadata.
        """
        if "transforms" not in data:
            _error(
                KeyError,
                "The dictionary does not contain the 'transforms' key.",
                _logger,
            )

        metadata = TransformMetadata()

        for transform in data["transforms"]:
            if "name" not in transform or "kwargs" not in transform:
                _error(
                    OscanaError,
                    "Failed to load the transforms for this data. Each "
                    "transform must have a 'name' and 'kwargs' key.",
                    _logger,
                )

                continue

            metadata.transforms.append((transform["name"], transform["kwargs"]))

        return metadata

    def print(self) -> None:
        """\
        Print the metadata.
        """
        print("Cuts & Transforms\n-----------------")
        for transform in self.transforms:
            type_ = "CUT" if transform[0].startswith("cut_") else "TFM"
            name = self._extract_transform_name(transform[0])
            kwargs = ", ".join(
                f"{key}={repr(value)}" for key, value in transform[1].items()
            )
            print(f"[{type_}] {name}({kwargs})")

        if not len(self.transforms):
            print("\t[ No Cuts & Transforms Applied ]")

    def __eq__(self, value: object) -> bool:
        """\
        Check if the metadata is the same for two datasets.

        Parameters
        ----------
        value : object
            The other object to compare with.

        Returns
        -------
        bool
            True if the metadata is the same, False otherwise.
        """
        # (1) Check if the other object is of the same type.

        if not isinstance(value, TransformMetadata):
            _error(
                ValueError,
                (
                    f"{self.__class__.__name__} can only be compared with "
                    "another instance of the same class."
                ),
                _logger,
            )

        # (2) Extract the transfrom names.

        these_names = set(
            [self._extract_transform_name(data[0]) for data in self.transforms]
        )
        other_names = set(
            [
                value._extract_transform_name(data[0])
                for data in value.transforms
            ]
        )

        return these_names == other_names

    def __ne__(self, value: object) -> bool:
        """\
        Check if the metadata is not the same for two datasets.

        Parameters
        ----------
        value : object
            The other object to compare with.

        Returns
        -------
        bool
            True if the metadata is not the same, False otherwise.
        """
        return not self.__eq__(value)

    def __str__(self) -> str:
        return (
            f"oscana.{self.__class__.__name__}"
            f"(n_transforms={len(self.transforms)})"
        )

    def __repr__(self) -> str:
        return str(self)
