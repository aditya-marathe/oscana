"""\
oscana / logger.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = ["init_root_logger"]

from typing import NoReturn

import platform, json, warnings
from importlib import resources
import logging, logging.config
from pathlib import Path

# ============================== [ Constants  ] ============================== #

# Note: A better way to do this would be using `importlib.resources.files` but
#       I am currenly using Python 3.8 which does not seem to have this feature.
#       The other option is to use `pkg_resources` but it is deprecated! So, for
#       now, I am using `importlib.resources.path` and going back two
#       directories to get the resources folder.

with resources.path("oscana", "") as _path:
    CONFIG_PATH = Path(_path).parent.parent / "res" / "configs"


_STACK_LEVEL: int = 3


# ========================== [ Warning Formatting ] ========================== #


def _warn_formatting(
    message: str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: str | None = None,
    line: int | None = None,
) -> str:
    """\
    [ Internal ] Formats the warnings in a specific way.
    
    Parameters
    ----------
    message : str
        The warning message.
        
    category : Warning
        The warning category.

    filename : str
        The filename where the warning was raised.

    lineno : int
        The line number where the warning was raised.

    file : str | None
        The file where the warning was raised.

    line : int | None
        The line number where the warning was raised.

    Returns
    -------
    str
        The formatted warning message.
    """
    return f"{category.__name__}: {message}\n"


# ============================ [ Implementation ] ============================ #

_is_root_logger_initialised: bool = False


def init_root_logger(
    logs_dir: str = "./",
    verbosity: str = "WARNING",
    config_file: str | None = None,
) -> None:
    """\
    Initialise the root logger.

    Parameters
    ----------
    logs_dir : str
        The directory where the logs should be stored.

    config_file : str, optional
        The path to the logging configuration file. If `None`, the default
        configuration file is used (recommended).

    Notes
    -----
    Usually, the default logging configuration file should be used (so leave the
    `config_file` parameter as `None`). If a custom configuration is used, make
    sure that this file is stored in JSON format.

    The credits for the logging configuration file goes to mCoding's video on
    YouTube: 
    
    https://www.youtube.com/watch?v=9L77QExPmI0&t=1118s&ab_channel=mCoding

    Thanks for making this tutorial!
    """
    global _is_root_logger_initialised  #  Acceptable use of `global` :P

    # A guard to ensure that the root logger is initialised only once.
    if _is_root_logger_initialised:
        logging.getLogger("Root").warning("Root logger already initialised!")
        return

    # Note: Some duplicated code from `oscana.utils._apply_wsl_prefix` is used
    #       here to avoid circular imports.
    #
    #       It pained me to do this, but I do not know a better way. If you know
    #       a better way, please let me know! Thanks!

    def _apply_wsl_prefix(dir_: str) -> Path:
        if platform.system() == "Linux" and dir_.startswith("C:"):
            dir_split = dir_.split("://")
            return Path(
                "/mnt/" + dir_split[0][0].lower() + "/" + dir_split[1]
            ).resolve()

        return Path(dir_).resolve()

    class OscanaFatalError(Exception):
        pass

    # (1) Get the logging configuration file.

    if config_file is None:
        config_file_resolved = (CONFIG_PATH / "logging.json").resolve()
    else:
        config_file_resolved = _apply_wsl_prefix(config_file).resolve()

    try:
        with open(config_file_resolved, "r") as file:
            config_dict = json.load(file)
    except FileNotFoundError:
        raise OscanaFatalError(
            f"Logging configuration file not found: {config_file_resolved}"
        )
    except json.JSONDecodeError:
        raise OscanaFatalError(
            "Invaild JSON format in logging configuration file."
        )

    # (2) Edit the directory of the log files.

    # TODO: This means that there is little flexibility in the configuration
    #       dictionary... Not sure how to fix this yet.

    config_dict["handlers"]["File"]["filename"] = (
        _apply_wsl_prefix(logs_dir)
        / config_dict["handlers"]["File"]["filename"]
    ).resolve()

    config_dict["handlers"]["ErrorFile"]["filename"] = (
        _apply_wsl_prefix(logs_dir)
        / config_dict["handlers"]["ErrorFile"]["filename"]
    ).resolve()

    config_dict["handlers"]["StdOut"]["level"] = verbosity

    logging.config.dictConfig(config=config_dict)

    # (3) Set the warning format.

    warnings.formatwarning = _warn_formatting

    # (4) Log the initialisation message.

    _is_root_logger_initialised = True
    logging.getLogger("Root").info("Logger initialised.")


def _error(
    error: type[Exception], message: str, logger: logging.Logger
) -> NoReturn:
    """\
    [ Internal ]

    Raises an error and logs the message.

    Parameters
    ----------
    error : type
        The error to raise.
    message : str
        The custom error message.
    """
    logger.error(f"{error.__name__}: {message}", stacklevel=_STACK_LEVEL)
    raise error(message)


def _warn(warning: type[Warning], message: str, logger: logging.Logger) -> None:
    """\
    [ Internal ]

    Raises a warning and logs the message.

    Parameters
    ----------
    warning : type
        The warning to raise.
    message : str
        The custom warning message.
    """
    logger.warning(f"{warning.__name__}: {message}", stacklevel=_STACK_LEVEL)
    warnings.warn(message, category=warning)
