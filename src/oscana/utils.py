"""\
oscana / utils.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = [
    "minos_numbers",
    "init_env_variables",
    "init_minos_numbers",
    "init_variable_search",
    "vs_search_variable",
    "vs_print_roots",
    "vs_print_variables",
    "destroy_variable_search",
]

from typing import TypeAlias, Any, Literal, Callable, Final

import os, platform, json, re
from importlib import import_module
import logging, logging.config
from pathlib import Path

import numpy as np
import numpy.typing as npt

import dotenv

from .logger import _error
from .errors import OscanaError
from .constants import RESOURCES_PATH

# =============================== [ Logging  ] =============================== #

_logger = logging.getLogger("Root")

# ======================= [ "Constant" Dictionaries  ] ======================= #

minos_numbers: Final[dict[str, Any]] = {}

_variable_hashtable: Final[dict[str, dict[str, list[str]]]] = {}

_supported_file_types: Final[tuple[str, ...]] = ("sntp_std",)

_DynamicFuncPrefix: TypeAlias = Literal["hlp_", "cut_", "tfm_"]

# ============================== [ Functions  ] ============================== #


def init_env_variables() -> None:
    """\
    Load the .env file in the root directory of the project.
    """
    if not dotenv.load_dotenv():
        _error(OscanaError, "Unsuccessful in loading the '.env' file!", _logger)

    _logger.info("Loaded environment variables from the '.env' file.")


def init_minos_numbers() -> None:
    """\
    Load the detector geometry and other numbers from a JSON file.
    """
    if minos_numbers:
        _logger.info("Already loaded MINOS numbers from the JSON file.")
        return

    file = (RESOURCES_PATH / "numbers.json").resolve()

    if not file.exists():
        _error(
            OscanaError,
            "MINOS numbers JSON file does not exist in Oscana resources!",
            _logger,
        )

    # minos_numbers.clear()  # Ensure that the dictionary is empty.

    with open(file, "r") as file:
        minos_numbers.update(json.load(file))

    _logger.info("Loaded MINOS numbers from the JSON file.")


def _apply_wsl_prefix(dir_: str) -> Path:
    """\
    [ Internal ]

    Apply the Windows Subsystem for Linux (WSL) prefix to the directory path.

    Parameters
    ----------
    dir_ : str
        The directory path.
    
    Returns
    -------
    Path
        The directory path with the WSL prefix.

    Notes
    -----
    This was added to make it convienent to work on WSL or Windows.
    """

    # Note: Only change the prefix from "C://" to "/mnt/" if the platform is
    #       Linux. This is done so that I can easily switch between WSL and
    #       Windows without needing to move files around.
    #
    #       This feature was only added for my convienence, and can be disabled.

    if platform.system() == "Linux" and dir_.startswith("C:"):
        dir_split = dir_.split("://")

        path = Path(
            "/mnt/" + dir_split[0][0].lower() + "/" + dir_split[1]
        ).resolve()

        _logger.debug("Applied WSL prefix '/mnt/' to a directory path.")

        return path

    return Path(dir_).resolve()


def _convert_from_utc(
    utc_timestamps: npt.NDArray[np.int_],
) -> npt.NDArray[np.datetime64]:
    """\
    [ Internal ]

    Convert a UTC timestamp to a datetime object.

    Parameters
    ----------
    utc_timestamp : int
        The UTC timestamp.

    Returns
    -------
    datetime
        The datetime object.
    """
    # This solution was thanks to ChatGPT - it actually works sometimes! :)
    return (
        np.datetime64("1970-01-01T00:00:00")
        + utc_timestamps.astype("timedelta64[s]")
    ).astype("O")


def _get_dir_from_env(file: str) -> Path:
    """\
    [ Internal ]

    Gets the full path of a file.

    Parameters
    ----------
    file : str
        The file name.

    Returns
    -------
    Path
        The full path of the file.
    """
    file_path = os.environ.get(file, None)

    if file_path is not None:

        _logger.debug(f"Found '{file}' in environment variables.")

        file_path_resolved = _apply_wsl_prefix(file_path).resolve()

        if file_path_resolved.exists():
            _logger.debug(f"Found '{file}' in the specified directory.")
            return file_path_resolved

        # Errors raised if file does not exist in the specified directory or...

        _error(
            FileNotFoundError,
            f"File '{file}' does not exist - check the '.env' file.",
            _logger,
        )

    # ... if the file does not exist in the '.env' file.
    _error(
        OscanaError,
        f"Reference to file '{file}' does not exist in the '.env' file.",
        _logger,
    )


# ======================= [ Oscana Dynamic Functions ] ======================= #

# Note: What the heck is a dynamic function? Well it's simply a function with a
#       version control system. How does that work? Simple. We set a function
#       naming convention and add in the date when that function was created
#       somewhere in its name.
#
#       As usual, use the Oscana function naming convention:
#
#           {hlp/cut/tfm}_{YYYYMMDD}_{function_name}
#
#       Then we initialise the function lookup dictionary and get the lookup
#       function using `get_func_lookup`. The lookup function can be used to get
#       the latest version of a function which you can specify by its name.

# Note: I do appreciate that this is not truly "dynamic" but you get the idea.


def get_func_lookup(
    globals_: dict[str, Any], prefix: _DynamicFuncPrefix
) -> Callable[[str], Callable[..., Any]]:
    """\
    Search for dynamic functions in the Oscana module.

    Parameters
    ----------
    globals : list
        The global variables.
    
    prefix : Literal["hlp_", "cut_", "tfm_"]
        The prefix of the dynamic functions.

    Returns
    -------
    Callable[[str], Callable[..., Any]]
        The "function lookup" function. :P
    """
    # Presistent dictionary to store relevant functions.
    funcs: dict[str, Callable[..., Any]] = {}

    for name, thing in globals_.items():
        if name.startswith(prefix):
            funcs[name] = thing  # All functions will have a different name.

    def func_lookup(func_name: str) -> Callable[..., Any]:
        """\
        Get the latest version of a dynamic function.

        Parameters
        ----------
        func_name : str
            The name of the function.

        Returns
        -------
        Callable[..., Any]
            Result of the function lookup.
        """
        result: list[str] = sorted(
            [name for name in funcs.keys() if name.endswith(func_name)]
        )

        if len(result):
            return funcs[result[-1]]  # Get the latest version of the function.

        _error(
            OscanaError,
            f"Unsuccessful in finding latest version of `{func_name}`.",
            _logger,
        )

    return func_lookup


# =============================== [ Plugins  ] =============================== #


def import_plugins(file: str) -> dict[str, Any]:
    """\
    Import all the plugins in the `plugins` directory.

    Parameters
    ----------


    Notes
    -----
    This function should be called in the `__init__.py` file module.
    """
    package_dir = Path(file).parent.resolve()
    plugins_dir = (Path(file).parent / "plugins").resolve()

    base_import_tree = f"oscana.{package_dir.name}.plugins"

    if package_dir.parent.name != "oscana":
        _error(
            OscanaError,
            (
                f"The '{package_dir.name}' package must be in 'oscana' for "
                f"plug-in support (not in '{package_dir.parent.name}'!)."
            ),
            _logger,
        )

    if not plugins_dir.exists():
        _error(
            FileNotFoundError,
            f"The 'plugins' directory does not exist in {Path(file).parent}.",
            _logger,
        )

    for module_file in plugins_dir.glob("*.py"):
        module = import_module(f"{base_import_tree}.{module_file.stem}")

        if hasattr(module, "__all__"):
            return {
                name: getattr(module, name)
                for name in getattr(module, "__all__")
            }

    return {}


# ========================= [ Variable Search Tool ] ========================= #


def init_variable_search() -> None:
    """\
    Initialise the variable search tool.
    """
    if _variable_hashtable:
        _logger.info("Already initialised the variable search tool.")
        return

    dir_ = (RESOURCES_PATH / "variables").resolve()

    if not dir_.exists():
        _error(
            OscanaError,
            "Variable search tool file does not exist in Oscana's resources!",
            _logger,
        )

    for f_name in _supported_file_types:
        f_dir = dir_ / f"{f_name}.txt"

        if not f_dir.exists():
            _error(
                FileNotFoundError,
                f"Variable search tool file '{f_name}.txt' is not supported.",
                _logger,
            )

        with open(f_dir, "r") as f:
            _variable_hashtable[f_name] = {}

            for line in f.readlines():
                line_split = line.strip().split(".")
                variable_root = line_split.pop(0)
                _variable_hashtable[f_name].setdefault(variable_root, [])

                if len(line_split):
                    _variable_hashtable[f_name][variable_root].append(
                        ".".join(line_split)
                    )

    _logger.info("Loaded variables to the search tool.")


def _vs_check_and_get(file_type: str) -> dict[str, list[str]]:
    """\
    [ Internal ]

    Check if VS was initialised and if the file type is supported.
    """
    if not _variable_hashtable:
        _error(
            OscanaError,
            "Variable search tool is not initialised.",
            _logger,
        )

    file_type_variables = _variable_hashtable.get(file_type, None)

    if file_type_variables is None:
        _error(
            OscanaError,
            f"File type '{file_type}' is not supported by the search tool.",
            _logger,
        )

    return file_type_variables


def vs_search_variable(file_type: str, query: str) -> None:
    """\
    Search for a variable using regular expressions black magic.
    """
    file_type_variables = _vs_check_and_get(file_type=file_type)

    query_compiled = re.compile(query)

    # Primitive, but it works.

    for key in sorted(file_type_variables.keys()):
        for var in file_type_variables[key]:
            if query_compiled.search(key + "." + var):
                print(key + "." + var)


def vs_print_roots(file_type: str) -> None:
    """\
    Variable Search Tool: Print the root variables for a given file type.

    Parameters
    ----------
    file_type : str
        The file type.
    """
    file_type_variables = _vs_check_and_get(file_type=file_type)

    for key in sorted(file_type_variables.keys()):
        print(key)


def vs_print_variables(file_type: str, root: str = "*") -> None:
    """\
    Variable Search Tool: Print the variables for a given file type.

    Parameters
    ----------
    file_type : str
        The file type.
    
    root : str
        The root variable. Default is '*'.
    """
    file_type_variables = _vs_check_and_get(file_type=file_type)

    print_title = lambda x: print(x + "\n" + "-" * len(x))

    if root == "*":
        for key in sorted(file_type_variables.keys()):
            print_title(key)

            for var in file_type_variables[key]:
                print(key + "." + var)

            print()

        return

    if root not in file_type_variables:
        _error(
            OscanaError,
            f"Root variable '{root}' is not found in the search tool.",
            _logger,
        )

    print_title(root)

    for var in file_type_variables[root]:
        print(root + "." + var)


def destroy_variable_search() -> None:
    """\
    Destroy the variable search tool.
    """
    _variable_hashtable.clear()
    _logger.info("Destroyed the variable search tool.")
