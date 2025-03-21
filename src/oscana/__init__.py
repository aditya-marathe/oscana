"""\
oscana

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

A Python package for the analysis of data from neutrino oscillation experiments.

Usage
-----

Simply import the package and it is mandatory to then initialise the logger.

```python
import oscana

# Initialise Oscana (Do this first!)
oscana.init()
```
"""

from __future__ import annotations

__version__ = "0.5.1"

from .constants import *
from .errors import *
from .images import *
from .logger import *
from .plotting import *
from .themes import *
from .utils import *

from . import data


def get_version() -> str:
    """\
    Get the version of the Oscana package.

    Returns
    -------
    str
        Version of the Oscana package.
    """
    return __version__


def print_version(fancy: bool = False) -> None:
    """\
    Print the version of the Oscana package.

    Parameters
    ----------
    fancy : bool, optional
        Whether to print the version in a fancy format. Defaults to `False`.
    """
    basic_text = f"Oscana (v{__version__})"

    if not fancy:
        return print(basic_text)

    text = f"{basic_text} - Neutrino Oscillation Analysis Package."

    print("\n+" + "-" * (len(text) + 2) + "+")
    print("| " + text + " |")
    print("+" + "-" * (len(text) + 2) + "+\n")


def init(
    logs_dir="./", verbosity="WARNING", config_file: str | None = None
) -> None:
    """\
    Initialise the Oscana package.
    
    Parameters
    ----------
    logs_dir : str, optional
        Directory to save the logs, by default "./"

    verbosity : str, optional
        Verbosity level for the logger, by default "WARNING"

    config_file : str, optional
        Path to the config file, by default None

    Notes
    -----
    You can also call Oscana `init_*` functions separately.
    """
    init_root_logger(
        logs_dir=logs_dir, verbosity=verbosity, config_file=config_file
    )
    init_env_variables()
    init_minos_numbers()


print_version(fancy=True)
