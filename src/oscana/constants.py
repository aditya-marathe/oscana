"""\
oscana / constants.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

This module contains all the constants used in the package.
"""

from __future__ import annotations

from typing import List, Literal, Iterator
from typing import Final, Union, Optional

__all__ = [
    # Package Constants
    "RESOURCES_PATH",
    # Data Types
    "IMAGE_DTYPE",
    "NUM_DTYPE",
    # SNTP Branches
    "SNTP_BR_STD",
    "SNTP_BR_BDL",
    "SNTP_BR_FIT",
    # SNTP Variable Collections
    "HEADER_VARIABLES",
    "IMAGE_BASIC_VARIABLES",
    "IMAGE_PE_VARIABLES",
    "IMAGE_SIGCOR_VARIABLES",
    "IMAGE_TIME_VARIABLES",
    "IMAGE_ALL_VARIABLES",
    "EVENT_VERTEX_VARIABLES",
    "MC_4MOMENTUM_VARIABLES",
    "MC_INTERACTION_VARIABLES",
    # Enums
    "EIAction",
    "EIResonance",
    "EIdHEP",
    "EInteraction",
    "EPlaneView",
]


from enum import Enum
from collections import defaultdict

from pathlib import Path
import importlib.resources as resources

import numpy as np

# =========================== [ Package Constants ] ========================== #


# Note: A better way to do this would be using `importlib.resources.files` but
#       I am currenly using Python 3.8 which does not seem to have this feature.
#       The other option is to use `pkg_resources` but it is deprecated! So, for
#       now, I am using `importlib.resources.path` and going back two
#       directories to get the resources folder.


with resources.path("oscana", "") as _path:
    RESOURCES_PATH = Path(_path).parent.parent / "res"


# ============================== [ Data Types ] ============================== #

IMAGE_DTYPE: Final = np.float32
NUM_DTYPE: Final = np.float32

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


class VariableCollection:
    """\
    VariableCollection
    ------------------

    A collection of ROOT SNTP variable names under a common ROOT branch.

    Attributes
    ----------
    list: list[str]
        The list of variable names in the collection.

    uproot : list[str]
        The list of variable names prefixed with the ROOT branch name.
    """

    __slots__ = ["_variables"]

    def __init__(
        self,
        variables: Optional[Union[VariableCollection, List[str]]] = None,
        root: str = "",
    ) -> None:
        """\
        Initialises a new `VariableCollection`.

        Parameters
        ----------
        variables : Optional[Union[VariableCollection, List[str]]]
            The list of variable names with their roots. Defaults to `None`.

        root : str
            The ROOT branch name to prefix to all variables in the collection.
        """
        # (1) Input validation
        if variables is None:
            if root:
                raise ValueError("Cannot add a root without passing variables!")

            variables = []
        elif isinstance(variables, list):
            if any("/" in string for string in variables) and root:
                raise ValueError(
                    "Cannot add a root when variables already have roots!"
                )

            if root:
                variables = [f"{root}/{var}" for var in variables]
        elif isinstance(variables, VariableCollection) and root:
            raise ValueError(
                "Cannot add a root when variables are already in a "
                "`VariableCollection`!"
            )

        # (2) Initialise the data structure
        self._variables = defaultdict(list)

        # (3) Add the variables (given at initialisation)
        self.add_variables(variables=variables)

    def add_variables(
        self, variables: Union[VariableCollection, List[str]]
    ) -> None:
        """\
        Adds variables to the collection.

        Parameters
        ----------
        variables : Union[VariableCollection, List[str]]
            The list of variables to add to the collection.
        """
        if isinstance(variables, VariableCollection):
            for branch, variables in variables._variables.items():
                self._variables[branch].extend(variables)

        elif isinstance(variables, list):
            for compound_variable in variables:
                split_result = compound_variable.split("/", 1)

                if len(split_result) == 2:
                    branch, variable = split_result
                elif len(split_result) == 1:
                    branch = ""
                    variable = split_result[0]
                else:
                    raise ValueError("Unreachable! Something went very wrong!")

                self._variables[branch].append(variable)

        else:
            raise ValueError(
                "The `variables` must be a `VariableCollection` or a list of "
                "variable names!"
            )

    @property
    def branch_names(self) -> List[str]:
        """\
        The ROOT branch name for the collection, if it exists.
        """
        return list(self._variables.keys())

    @property
    def uproot(self) -> List[str]:
        """\
        Gets the list of variable names prefixed with the ROOT branch name.

        Returns
        -------
        list[str]
            The list of variable names prefixed with the ROOT branch name.
        """
        return [
            f"{branch}/{variable}"
            for branch, variables in self._variables.items()
            if branch
            for variable in variables
        ]

    @property
    def list_(self) -> List[str]:
        """\
        Gets the list of variable names in the collection.

        Returns
        -------
        list[str]
            The list of variable names in the collection.
        """
        return [
            variable
            for variables in self._variables.values()
            for variable in variables
        ]

    # List-like Behaviours

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_)

    def __add__(self, other: object) -> VariableCollection:
        this_variables = self.uproot

        if isinstance(other, VariableCollection):
            return VariableCollection(variables=this_variables + other.uproot)

        elif isinstance(other, list):
            return VariableCollection(variables=this_variables + other)

        raise ValueError(
            "Operation `+` only supported between `VariableCollections` and "
            "lists!"
        )

    def __radd__(self, other: object) -> "VariableCollection":
        return self.__add__(other=other)

    def __mul__(self, value: object) -> "VariableCollection":
        raise ValueError(
            "Operation `*` not supported for `VariableCollections`!"
        )

    def __rmul__(self, value: object) -> "VariableCollection":
        return self.__mul__(value=value)

    def __getitem__(self, index: int) -> str:
        return self.list_[index]

    def __len__(self) -> int:
        return len(self.list_)

    def __str__(self) -> str:
        return f"VariableCollection({self.list_})"

    def __repr__(self) -> str:
        return str(self.list_)


HEADER_VARIABLES: Final[VariableCollection] = VariableCollection(
    variables=[
        "fRun",  # Run number
        "fSubRun",  # Subrun number
        "fSnarl",  # Snarl number
        "fEvent",  # Event number
    ],
    root=SNTP_BR_STD,
)

IMAGE_BASIC_VARIABLES: Final[VariableCollection] = VariableCollection(
    variables=[
        "stp.planeview",  # Plane view
        "stp.strip",  # Strip number
        "stp.plane",  # Plane number
    ],
    root=SNTP_BR_STD,
)

IMAGE_PE_VARIABLES: Final[VariableCollection] = VariableCollection(
    variables=[
        "stp.ph0.pe",  # Photoelectrons (East)
        "stp.ph1.pe",  # Photoelectrons (West)
    ],
    root=SNTP_BR_STD,
)

IMAGE_SIGCOR_VARIABLES: Final[VariableCollection] = VariableCollection(
    variables=[
        "stp.ph0.sigcor",  # Normalised strip response (East)
        "stp.ph1.sigcor",  # Normalised strip response (West)
    ],
    root=SNTP_BR_STD,
)

IMAGE_TIME_VARIABLES: Final[VariableCollection] = VariableCollection(
    variables=[
        "stp.time0",  # Charge weighted mean time [s] (East)
        "stp.time1",  # Charge weighted mean time [s] (West)
    ],
    root=SNTP_BR_STD,
)

IMAGE_ALL_VARIABLES: Final[VariableCollection] = (
    IMAGE_BASIC_VARIABLES
    + IMAGE_PE_VARIABLES
    + IMAGE_SIGCOR_VARIABLES
    + IMAGE_TIME_VARIABLES
)

EVENT_VERTEX_VARIABLES: Final[VariableCollection] = VariableCollection(
    variables=[
        "evt.vtx.x",  # Event vertex x-coordinate [m]
        "evt.vtx.y",  # Event vertex y-coordinate [m]
        "evt.vtx.z",  # Event vertex z-coordinate [m]
    ],
    root=SNTP_BR_STD,
)

MC_4MOMENTUM_VARIABLES: Final[VariableCollection] = VariableCollection(
    variables=[
        "mc.p4neunoosc[4]",  # Neutrino 4-momentum
        "mc.p4mu1[4]",  # Primary muon 4-momentum
        "mc.p4shw[4]",  # Hadronic shower 4-momentum
    ],
    root=SNTP_BR_STD,
)

MC_INTERACTION_VARIABLES: Final[VariableCollection] = VariableCollection(
    variables=[
        "mc.iaction",  # Interaction type (CC / NC)
        "mc.inunoosc",  # Interacting neutrino PDG code
    ],
    root=SNTP_BR_STD,
)

# ================================ [ Enums  ] ================================ #


class _BaseEnum(Enum):
    """\
    [ Internal ] Base class for all Enums in the package.
    """

    @property
    def latex(self) -> str:
        return "< ? >"

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()

    def __repr__(self) -> str:
        return f"oscana.{self.__class__.__name__}.{self.name}"


class EIAction(_BaseEnum):
    """\
    Neutrino Interaction Types (Neutral / Charged Current)
    """

    NC = 0
    CC = 1
    UNKNOWN = -1

    @classmethod
    def _missing_(cls, value: object) -> EIAction:
        return cls(cls.UNKNOWN)


class EIResonance(_BaseEnum):
    """\
    Neutrino Interaction Types (Resonance)
    """

    QES = 1001  # Quasi-Elastic Scattering
    RES = 1002  # Resonance Production
    DIS = 1003  # Deep Inelastic Scattering
    COH = 1004  # Coherent Pion Production
    IMD = 1005  # Inverse Muon Decay


class EIdHEP(_BaseEnum):
    """\
    Particle ID Codes (PDG Codes)
    """

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


class EInteraction(_BaseEnum):
    """\
    Interaction Codes

    Note
    ----
    The interaction codes are calculated by multiplying the "mc.iaction" and
    "stdhep.IdHEP" columns in the data. The "mc.iaction" is the true interaction 
     (either CC or NC) and the "stdhep.IdHEP" column is the interacting neutrino
     (NuE, NuMu, NuTau, or their anti-particles).
    """

    # CC Interactions
    NUECC = EIdHEP.ELECTRON_NU.value
    NUMUCC = EIdHEP.MUON_NU.value
    NUTAUCC = EIdHEP.TAU_NU.value

    ANTINUECC = -EIdHEP.ELECTRON_NU.value
    ANTINUMUCC = -EIdHEP.MUON_NU.value
    ANTINUTAUCC = -EIdHEP.TAU_NU.value

    # NC Interactions
    NC = 0

    # Unknown
    UNKNOWN = -1

    @property
    def latex(self) -> str:
        return {
            self.NUECC: r"$\nu_\text{e}$ CC",
            self.NUMUCC: r"$\nu_\mu$ CC",
            self.NUTAUCC: r"$\nu_\tau$ CC",
            self.ANTINUECC: r"$\bar{\nu}_e$ CC",
            self.ANTINUMUCC: r"$\bar{\nu}_\mu$ CC",
            self.ANTINUTAUCC: r"$\bar{\nu}_\tau$ CC",
            self.NC: r"NC",
        }.get(self, r"< ? >")

    @classmethod
    def _missing_(cls, value: object) -> EInteraction:
        return cls(cls.UNKNOWN)


class ESimpleInteraction(_BaseEnum):
    """\
    Simplified Interaction Codes

    Note
    ----
    The simplified interaction codes are calculated by taking the absolute
    value of the product of "mc.iaction" and "stdhep.IdHEP". The "mc.iaction"
    is the true interaction (either CC or NC) and the "stdhep.IdHEP" column is
    the interacting neutrino (NuE, NuMu, NuTau, or their anti-particles).
    """

    # CC Interactions
    NUECC = abs(EIdHEP.ELECTRON_NU.value)
    NUMUCC = abs(EIdHEP.MUON_NU.value)
    NUTAUCC = abs(EIdHEP.TAU_NU.value)

    # NC Interactions
    NC = 0

    # Unknown
    UNKNOWN = -1

    @property
    def latex(self) -> str:
        return {
            self.NUECC: r"$\stackrel{(\rule{0.8em}{0.4pt})}{\nu}_\text{e}$ CC",
            self.NUMUCC: r"$\stackrel{(\rule{0.8em}{0.4pt})}{\nu}_\mu$ CC",
            self.NUTAUCC: r"$\stackrel{(\rule{0.8em}{0.4pt})}{\nu}_\tau$ CC",
            self.NC: r"NC",
        }.get(self, r"< ? >")

    @classmethod
    def _missing_(cls, value: object) -> ESimpleInteraction:
        return cls(cls.UNKNOWN)


class EPlaneView(_BaseEnum):
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
