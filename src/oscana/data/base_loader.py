"""\
oscana / data / base_loader.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

Stuff in this module is no longer used and is saved for future reference ONLY.
"""

from typing import Callable

from abc import ABC, abstractmethod


class DataLoaderBase(ABC):
    """\
    [ Internal ]
    
    Base class for all data loaders.

    General Philosophy
    ------------------

    -   All data loaders must inherit from this class and define its abstract 
        methods.
    
    -   The loaded data must be stored in the `self._data` variable.

    -   When importing, espciailly from a SNTP file, we should not be loading
        all they keys, instead we should only load the keys that are required.
        The keys should be stored in the `self._variables` variable.

    -   Must include the `self._metadata` variable. This will contain the 
        individual metadata for each file and the metadata must be stored in a 
        Python `dataclass`.

    -   Must include the `self._transforms` variable, which is a list of names 
        of functions which transform the data. It is crucial that we keep track
        of the transformations applied to the data.
    """

    @abstractmethod
    def load_from_files(self, files: list[str]) -> None:
        # Note: Only the environment variable names should be passed here, not
        #       the full file path.
        pass

    @abstractmethod
    def transform(self, transforms: list[Callable]) -> None:
        # Note: Transformations will be applied in the order they are passed.
        pass

    @abstractmethod
    def copy(self) -> "DataLoaderBase":
        # Note: I do not yet see a universe where this would be used, but it is
        #       good to have it there.
        pass

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
