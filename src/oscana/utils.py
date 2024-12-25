"""\
oscana / utils.py
"""

__all__ = [
    "OscanaError",
    "load_env_file",
    "apply_wsl_prefix",
]

import platform
from pathlib import Path

import dotenv

# ============================== [ Exceptions ] ============================== #


class OscanaError(Exception):
    pass


# ============================== [ Functions  ] ============================== #


def load_env_file() -> None:
    """\
    Load the .env file in the root directory of the project.
    """
    if not dotenv.load_dotenv():
        raise OscanaError("Unsuccessful in loading .env file!")


def apply_wsl_prefix(dir_: str) -> Path:
    """\
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
        return Path("/mnt/" + dir_split[0][0].lower() + "/" + dir_split[1])

    return Path(dir_)
