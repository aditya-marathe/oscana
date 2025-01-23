"""\
oscana / utils.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

__all__ = [
    "OscanaError",
    "init_env_variables",
    "_get_dir_from_env",
]

import logging.config
import os, platform, json
from importlib import resources
from pathlib import Path

import dotenv

# ============================== [ Constants  ] ============================== #

# Note: A better way to do this would be using `importlib.resources.files` but
#       I am currenly using Python 3.8 which does not seem to have this feature.
#       The other option is to use `pkg_resources` but it is deprecated! So, for
#       now, I am using `importlib.resources.path` and going back two
#       directories to get the resources folder.

with resources.path("oscana", "") as _path:
    CONFIG_PATH = Path(_path) / "configs"

# ============================== [ Exceptions ] ============================== #


class OscanaError(Exception):
    pass


# ============================== [ Functions  ] ============================== #


def init_env_variables() -> None:
    """\
    Load the .env file in the root directory of the project.
    """
    if not dotenv.load_dotenv():
        raise OscanaError("Unsuccessful in loading .env file!")


def init_root_logger(config_file: str | None = None) -> None:
    """\
    Initialise the root logger.

    Notes
    -----
    Usually, the default logging configuration file should be used (so leave the
    `config_file` parameter as `None`). If a custom configuration is used, make
    sure that this file is stored in JSON format.
    """
    raise NotImplementedError

    # Work in Progress

    if config_file is None:
        config_file_resolved = (CONFIG_PATH / "logging.json").resolve()
    else:
        config_file_resolved = _apply_wsl_prefix(config_file).resolve()

    with open(config_file_resolved, "r") as file:
        logging.config.dictConfig(config=json.load(file))


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

    # Only change the prefix if the platform is Linux.
    if platform.system() == "Linux":
        dir_split = dir_.split("://")
        return Path(
            "/mnt/" + dir_split[0][0].lower() + "/" + dir_split[1]
        ).resolve()

    return Path(dir_).resolve()


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

        file_path_resolved = _apply_wsl_prefix(file_path).resolve()

        if file_path_resolved.exists():
            return file_path_resolved

    raise OscanaError(f"File '{file}' does not exist!")
