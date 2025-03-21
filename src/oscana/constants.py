"""\
oscana / constants.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

This module contains all the constants used in the package.
"""

from __future__ import annotations

__all__ = [
    # Package Constants
    "RESOURCES_PATH",
    # SNTP Branches
    "SNTP_BR_STD",
    "SNTP_BR_BDL",
    "SNTP_BR_FIT",
    # SNTP Variable Collections
    "IMAGE_VARIABLES",
    # Enums
    "EIAction",
    "EIResonance",
    "EIdHEP",
    "EPlaneView",
]

from typing import Final, Literal
from pathlib import Path
import importlib.resources as resources
from enum import Enum

# =========================== [ Package Constants ] ========================== #


# Note: A better way to do this would be using `importlib.resources.files` but
#       I am currenly using Python 3.8 which does not seem to have this feature.
#       The other option is to use `pkg_resources` but it is deprecated! So, for
#       now, I am using `importlib.resources.path` and going back two
#       directories to get the resources folder.


with resources.path("oscana", "") as _path:
    RESOURCES_PATH = Path(_path).parent.parent / "res"


# ============================ [ SNTP Branches  ] ============================ #

SNTP_BR_STD: Final[Literal["NtpSt"]] = "NtpSt"
SNTP_BR_BDL: Final[Literal["NtpBDLite"]] = "NtpBDLite"
SNTP_BR_FIT: Final[Literal["NtpFitSA"]] = "NtpFitSA"

# ============================ [ SNTP Variables ] ============================ #

SNTP_VR_DETECTOR: Final[str] = (
    "NtpStRecord/RecRecordImp<RecCandHeader>/fHeader.RecPhysicsHeader/"
    "fHeader.RecDataHeader/fHeader.RecHeader/fHeader.fVldContext.fDetector"
)
SNTP_VR_SIM: Final[str] = (
    "NtpStRecord/RecRecordImp<RecCandHeader>/fHeader.RecPhysicsHeader"
    "/fHeader.RecDataHeader/fHeader.RecHeader/fHeader.fVldContext.fSimFlag"
)
SNTP_VR_RUN: Final[str] = (
    "NtpStRecord/RecRecordImp<RecCandHeader>/fHeader.RecPhysicsHeader"
    "/fHeader.RecDataHeader/fHeader.fRun"
)
SNTP_VR_EVT_UTC: Final[str] = (
    "NtpStRecord/RecRecordImp<RecCandHeader>/fHeader.RecPhysicsHeader/"
    "fHeader.RecDataHeader/fHeader.RecHeader/"
    "fHeader.fVldContext.fTimeStamp.fSec"
)

# ====================== [ SNTP Variable Collections  ] ====================== #

IMAGE_VARIABLES: Final[list[str]] = [
    f"{SNTP_BR_STD}/stp.planeview",
    f"{SNTP_BR_STD}/stp.strip",
    f"{SNTP_BR_STD}/stp.plane",
    f"{SNTP_BR_STD}/stp.ph0.pe",
    f"{SNTP_BR_STD}/stp.ph1.pe",
]

# ================================ [ Enums  ] ================================ #


class BaseEnum(Enum):
    """\
    [ Internal ] Base class for all Enums in the package.
    """

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()

    def __repr__(self) -> str:
        return f"Oscana.{self.__class__.__name__}.{self.name}"


class EIAction(BaseEnum):
    NC = 0
    CC = 1
    UNKNOWN = -1

    @classmethod
    def _missing_(cls, value: object) -> EIAction:
        return cls(cls.UNKNOWN)


class EIResonance(BaseEnum):
    QES = 1001  # Quasi-Elastic Scattering
    RES = 1002  # Resonance Production
    DIS = 1003  # Deep Inelastic Scattering
    COH = 1004  # Coherent Pion Production
    IMD = 1005  # Inverse Muon Decay


class EIdHEP(BaseEnum):
    PHOTON = 22
    ELECTRON = 11
    MUON = 13
    TAU = 15
    ELECTRON_NU = 12
    MUON_NU = 14
    TAU_NU = 16
    CHARGED_PION = 211
    NEUTRAL_PION = 111
    ETA = 221
    CHARGED_RHO = 213
    NEUTRAL_RHO = 113
    OMEGA = 223
    CHARGED_KAON = 321
    NEUTRAL_KAON = 311
    K_SHORT = 310
    K_LONG = 130
    PROTON = 2212
    NEUTRON = 2112
    DELTA_MINUS = 1114
    DELTA_ZERO = 2114
    DELTA_PLUS = 2214
    DELTA_PLUS_PLUS = 2224
    GEANTINO = 28  # Placeholder
    UNKNOWN = -1

    @classmethod
    def _missing_(cls, value: object) -> EIdHEP:
        return cls(cls.UNKNOWN)


class EPlaneView(BaseEnum):
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
    def _missing_(cls, value: object) -> EPlaneView:
        return cls(cls.Unknown)
