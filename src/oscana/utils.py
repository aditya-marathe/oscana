"""\
oscana / utils.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = [
    "SNTP_BR_STD",
    "SNTP_BR_BDL",
    "SNTP_BR_FIT",
    "DetectorEnum",
    "SimFlagEnum",
    # "FileMetadataEnum",  # --> This is going to get yeeted soon.
    "PlaneView",
    "OscanaError",
    "init_env_variables",
    "init_minos_numbers",
]

from typing import Any

import os, platform, json
from importlib import resources
import logging, logging.config
from pathlib import Path
from enum import Enum

import numpy as np
import numpy.typing as npt

import dotenv

from .logger import _error

# =============================== [ Logging  ] =============================== #

logger = logging.getLogger("Root")

# ========================= [ Constants and Enums  ] ========================= #

# Note: A better way to do this would be using `importlib.resources.files` but
#       I am currenly using Python 3.8 which does not seem to have this feature.
#       The other option is to use `pkg_resources` but it is deprecated! So, for
#       now, I am using `importlib.resources.path` and going back two
#       directories to get the resources folder.

with resources.path("oscana", "") as _path:
    RESOURCES_PATH = Path(_path).parent.parent / "res"

# Note: This is very loosely, a constant. Seriously, do not change this!
minos_numbers: dict[str, Any] = {}

# SNTP Branches
SNTP_BR_STD = "NtpSt"
SNTP_BR_BDL = "NtpBDLite"
SNTP_BR_FIT = "NtpFitSA"

# SNTP Leaf Variables
SNTP_VR_DETECTOR = (
    "NtpStRecord/RecRecordImp<RecCandHeader>/fHeader.RecPhysicsHeader/"
    "fHeader.RecDataHeader/fHeader.RecHeader/fHeader.fVldContext.fDetector"
)
SNTP_VR_SIM = (
    "NtpStRecord/RecRecordImp<RecCandHeader>/fHeader.RecPhysicsHeader"
    "/fHeader.RecDataHeader/fHeader.RecHeader/fHeader.fVldContext.fSimFlag"
)
SNTP_VR_RUN = (
    "NtpStRecord/RecRecordImp<RecCandHeader>/fHeader.RecPhysicsHeader"
    "/fHeader.RecDataHeader/fHeader.fRun"
)
SNTP_VR_EVT_UTC = (
    "NtpStRecord/RecRecordImp<RecCandHeader>/fHeader.RecPhysicsHeader/"
    "fHeader.RecDataHeader/fHeader.RecHeader/"
    "fHeader.fVldContext.fTimeStamp.fSec"
)


class DetectorEnum(Enum):
    Near = 1
    Far = 2
    Unknown = -1

    @classmethod
    def _missing_(cls, value: object) -> DetectorEnum:
        return cls(cls.Unknown)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Oscana.{self.__class__.__name__}.{self.name}"


class SimFlagEnum(Enum):
    Data = 0
    DaqFakeData = 1
    MC = 2
    Reroot = 4
    Unknown = 8

    @classmethod
    def _missing_(cls, value: object) -> SimFlagEnum:
        return cls(cls.Unknown)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Oscana.{self.__class__.__name__}.{self.name}"


class FileMetadataEnum(Enum):
    # Neutrino Source
    Beam = "Beam"
    Atmospheric = "Atmospheric"
    # Unknown
    Unknown = "Unknown"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Oscana.{self.__class__.__name__}.{self.name}"


class PlaneView(Enum):
    # Note: I am using (mostly) the same Enum as the MINOS code (refer to
    #       `EPlaneView` on Doxygen).

    # Standard
    X = 0
    Y = 1
    U = 2
    V = 3

    # Calibration Detector
    A = 4
    B = 5

    # Veto Shield
    TopFlat = 8
    TopESlant = 9
    TopWSlant = 10
    WallOnEdge = 11
    WallESlant = 12
    WallWSlant = 13

    # Unknown
    Unknown = 7  # --> For some reason this is a 7 and not a 6?

    @classmethod
    def _missing_(cls, value: object) -> PlaneView:
        return cls(cls.Unknown)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Oscana.{self.__class__.__name__}.{self.name}"


# ============================== [ Exceptions ] ============================== #


class OscanaError(Exception):
    pass


# ============================== [ Functions  ] ============================== #


def init_env_variables() -> None:
    """\
    Load the .env file in the root directory of the project.
    """
    if not dotenv.load_dotenv():
        _error(OscanaError, "Unsuccessful in loading the '.env' file!", logger)

    logger.info("Loaded environment variables from the '.env' file.")


def init_minos_numbers() -> None:
    """\
    Load the detector geometry and other numbers from a JSON file.
    """
    file = (RESOURCES_PATH / "numbers.json").resolve()

    if not file.exists():
        _error(
            OscanaError,
            "MINOS numbers JSON file does not exist in Oscana resources!",
            logger,
        )

    minos_numbers.clear()  # Ensure that the dictionary is empty.

    with open(file, "r") as file:
        minos_numbers.update(json.load(file))

    logger.info("Loaded MINOS numbers from the JSON file.")


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

    if platform.system() == "Linux":
        dir_split = dir_.split("://")

        path = Path(
            "/mnt/" + dir_split[0][0].lower() + "/" + dir_split[1]
        ).resolve()

        logger.debug("Applied WSL prefix '/mnt/' to a directory path.")

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

        logger.debug(f"Found '{file}' in environment variables.")

        file_path_resolved = _apply_wsl_prefix(file_path).resolve()

        if file_path_resolved.exists():
            logger.debug(f"Found '{file}' in the specified directory.")
            return file_path_resolved

        # Errors raised if file does not exist in the specified directory or...

        _error(
            FileNotFoundError,
            f"File '{file}' does not exist - check the '.env' file.",
            logger,
        )

    # ... if the file does not exist in the '.env' file.
    _error(
        OscanaError,
        f"Reference to file '{file}' does not exist in the '.env' file.",
        logger,
    )
