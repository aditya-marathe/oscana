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

# Note: Version numbers now (30/10/2025) follow the Semantic Versioning scheme:
#
#           "{MAJOR}.{MINOR}.{PATCH}"
#

__version__ = "5.0.0"

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
    from .escape import Style

    description = "Neutrino Oscillation Analysis Package."
    basic_text = f"Oscana (v{__version__}) - {description}"

    if not fancy:
        return print(basic_text)

    fancy_text = (
        Style.BD
        + (Style.FG[255] + Style.BG[208] + " ")
        + (Style.FG[255] + Style.BG[208] + "O")
        + (Style.FG[255] + Style.BG[209] + "s")
        + (Style.FG[255] + Style.BG[210] + "c")
        + (Style.FG[255] + Style.BG[211] + "a")
        + (Style.FG[255] + Style.BG[212] + "n")
        + (Style.FG[255] + Style.BG[213] + "a")
        + (Style.FG[255] + Style.BG[213] + " ")
        + Style.R
        + Style.BD
        + f" (v{__version__})"
        + Style.R
    )

    fancy_text = f"{fancy_text} - {Style.IT + description + Style.R}"

    print("\n+" + "-" * (len(basic_text) + 2 + 2) + "+")
    print("| " + fancy_text + " |")
    print("+" + "-" * (len(basic_text) + 2 + 2) + "+\n")


def init(
    logs_dir="./",
    verbosity="WARNING",
    config_file: str | None = None,
    dotenv_dir: str | None = None,
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

    dotenv_dir : str | Path | None, optional
        The directory of the .env file. Defaults to `None`.

    Notes
    -----
    You can also call Oscana `init_*` functions separately.
    """
    init_root_logger(
        logs_dir=logs_dir, verbosity=verbosity, config_file=config_file
    )
    init_env_variables(dotenv_dir=dotenv_dir)
    init_minos_numbers()


print_version(fancy=True)
