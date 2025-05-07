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

import logging
from dataclasses import dataclass, field

from ..logger import _error

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

    cuts: list[str] = field(default_factory=list)
    transforms: list[str] = field(default_factory=list)

    def _add_transform(self, name: str) -> None:
        """\
        Add a new transform to the metadata.

        Parameters
        ----------
        name : str
            Name of the transform.

        Notes
        -----
        The function names must follow the Oscana naming convention:
            {`cut` / `tfm`}_{YYYYMMDD}_{name}
        """
        # Note: Here I am not checking if the transform has already been added!
        #       This will be taken care of by the `DataHandler` class.

        if name.startswith("cut_"):
            self.cuts.append(name)
        elif name.startswith("tfm_"):
            self.transforms.append(name)
        else:
            _error(
                ValueError,
                (
                    f"Transform name '{name}' does not adhere to Oscana naming "
                    "convention."
                ),
                _logger,
            )

    def print(self) -> None:
        """\
        Print the metadata.
        """
        print("Cuts\n----")
        for cut in self.cuts:
            print(f"\t- {cut}")

        if not len(self.cuts):
            print("\t[ No Cuts Applied ]")

        print("\nTransforms\n----------")
        for transform in self.transforms:
            print(f"\t- {transform}")

        if not len(self.transforms):
            print("\t[ No Transforms Applied ]")

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TransformMetadata):
            _error(
                ValueError,
                (
                    f"{self.__class__.__name__} can only be compared with "
                    "another instance of the same class."
                ),
                _logger,
            )

        # Check if the same cuts and transforms have been applied.
        if (set(self.cuts) != set(value.cuts)) and (
            set(self.transforms) != set(value.transforms)
        ):
            return False

        return True

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __str__(self) -> str:
        return (
            f"Oscana.{self.__class__.__name__}(n_cuts={len(self.cuts)}, "
            f"n_transforms={len(self.transforms)})"
        )

    def __repr__(self) -> str:
        return str(self)
