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

# Initialise Oscana
oscana.init_root_logger()  # --> Do this first!
oscana.init_env_variables()
```
"""

from __future__ import annotations

__version__ = "0.5.0"

from .utils import *

from .logger import init_root_logger

from .data import *

from .evd import *


print(f"Oscana (v{__version__}) - Neutrino Oscillation Analysis Package.")


def get_version() -> str:
    """\
    Get the version of the Oscana package.

    Returns
    -------
    str
        Version of the Oscana package.
    """
    return __version__


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
    """
    init_root_logger(
        logs_dir=logs_dir, verbosity=verbosity, config_file=config_file
    )
    init_env_variables()
