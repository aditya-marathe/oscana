"""\
oscana / data / enumerations.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = []

from enum import auto

from ..utils import BaseEnum


class EFileFormat(BaseEnum):
    """\
    [ Internal ]

    File Formats
    ------------
        - ROOT    : ROOT files.
        - HDF5    : HDF5 files.
        - UNKNOWN : Unknown file format.
    """

    ROOT = "root"
    HDF5 = "h5"
    UNKNOWN = "????"

    @classmethod
    def _missing_(cls, value: object) -> EFileFormat:
        return cls.UNKNOWN


class EFileType(BaseEnum):
    """\
    [ Internal ]

    File Types
    ----------
        - DATA        : Experiment data.
        - MONTE_CARLO : Monte Carlo data.
        - UNKNOWN     : Unknown file type.
    """

    DATA = auto()
    MONTE_CARLO = auto()
    UNKNOWN = auto()

    @classmethod
    def _missing_(cls, value: object) -> EFileType:
        return cls.UNKNOWN


class ESimFlag(BaseEnum):
    """\
    [ Internal ]

    Simulation Flags
    ----------------
        - DATA          : Experiment data.
        - DAQ_FAKE_DATA : DAQ fake data.
        - MC            : Monte Carlo data.
        - REROOT        : Re-rooted data.
        - UNKNOWN       : Unknown simulation flag.
    """

    DATA = 0
    DAQ_FAKE_DATA = 1
    MONTE_CARLO = 2
    REROOT = 4
    UNKNOWN = 8

    @classmethod
    def _missing_(cls, value: object) -> ESimFlag:
        return cls.UNKNOWN


class EExperiment(BaseEnum):
    """\
    [ Internal ]

    Experiments
    -----------
        - MINOS     : MINOS experiment.
        - MINOS_PLUS : MINOS+ experiment.
        - NOVA      : NOvA experiment.
        - UNKNOWN   : Unknown experiment.
    """

    MINOS = auto()
    MINOS_PLUS = auto()
    NOVA = auto()
    UNKNOWN = auto()

    @classmethod
    def _missing_(cls, value: object) -> EExperiment:
        return cls.UNKNOWN


class EDetector(BaseEnum):
    """\
    [ Internal ]

    Detectors
    ---------
        - CALIBRATION : Calibration detector (aka. CalDet).
        - NEAR        : Near detector.
        - FAR         : Far detector.
        - UNKNOWN     : Unknown detector.
    """

    CALIBRATION = 0
    NEAR = 1
    FAR = 2
    UNKNOWN = -1

    @classmethod
    def _missing_(cls, value: object) -> EDetector:
        return cls.UNKNOWN


class EMCVersion(BaseEnum):
    """\
    [ Internal ]

    MC Versions
    -----------
        - AVOCADO : Avocado.
        - BEET    : Beet.
        - CARROT  : Carrot 6 & 8.
        - DAIKON  : Daikon 0 to 11.
        - UNKNOWN : Unknown version.
    """

    AVOCADO = "A"
    BEET = "B"
    CARROT = "C"
    DAIKON = "D"
    UNKNOWN = "?"

    @classmethod
    def _missing_(cls, value: object) -> EMCVersion:
        return cls.UNKNOWN


class ERecoVersion(BaseEnum):
    """\
    [ Internal ]

    Reconstruction Versions
    -----------------------
        - ASH     : Ash.      [Release 1.18.0]
        - BIRCH   : Birch.    [Release 1.18.2]
        - CEDAR   : Cedar.    [Release 1.24.0]
        - DOGWOOD : Dogwood.  [Release 2.00.0]
        - UNKNOWN : Unknown version.
    """

    ASH = "ash"
    BIRCH = "birch"
    CEDAR = "cedar"
    DOGWOOD = "dogwood"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> ERecoVersion:
        return cls.UNKNOWN


class EHornPosition(BaseEnum):
    """\
    [ Internal ]

    Horn Positions
    --------------
        - LOW_ENERGY   : Low energy horn.
        - HIGH_ENERGY  : High energy horn.
        - MEDIUM_ENERGY: Medium energy horn.
        - UNKNOWN      : Unknown horn position.
    """

    LOW_ENERGY = "L"
    HIGH_ENERGY = "H"
    MEDIUM_ENERGY = "M"
    UNKNOWN = "?"

    @classmethod
    def _missing_(cls, value: object) -> EHornPosition:
        return cls.UNKNOWN


class EHornCurrent(BaseEnum):
    """\
    [ Internal ]

    Current Signs
    -------------
        - FORWARD : Forward horn current.
        - REVERSE : Reverse horn current.
        - UNKNOWN : Unknown current sign.
"""

    FORWARD = "N"
    REVERSE = "R"
    UNKNOWN = "?"

    @classmethod
    def _missing_(cls, value: object) -> EHornCurrent:
        return cls.UNKNOWN


# Daikon Specific


class EDaikonIntRegion(BaseEnum):
    """\
    [ Internal ]

    Interaction Regions
    -------------------
        - ATMOSPHERE        : In the atmosphere only.
        - DETECTOR          : In the detector only.
        - ROCK              : In the rock only.
        - DETECTOR_ROCK     : In the detector & rock overlayed.
        - DETECTOR_FIDUCIAL : In the detector fiducial volume only.
        - SMALL_FIDUCIAL    : In "smallfid" only.
        - UNKNOWN           : Unknown region.

    Notes
    -----
    Compatible with Daikon and Beet MC versions only!
    """

    ATMOSPHERE = 0
    DETECTOR = 1
    ROCK = 2
    DETECTOR_ROCK = 3
    DETECTOR_FIDUCIAL = 4
    SMALL_FIDUCIAL = 5  # No idea what this one is...
    UNKNOWN = -1

    @classmethod
    def _missing_(cls, value: object) -> EDaikonIntRegion:
        return cls.UNKNOWN


class EDaikonFlavour(BaseEnum):
    """\
    [ Internal ]

    Flavours
    --------
        - UNOSCILLTAED         : Unoscillated.
        - NU_E                 : Electron neutrino.
        - NU_TAU               : Tau neutrino.
        - INVERTED_BEAM        : Inverted beam (NuE -> NuMu; NuMu -> NuE).
        - FAR_OSCILLATED_MOCK  : Far oscillated mock.
        - UNKNOWN              : Unknown flavour.
    """

    UNOSCILLATED = 0
    NU_E = 1
    NU_TAU = 3
    INVERTED_BEAM = 4
    FAR_OSCILLATED_MOCK = 9
    UNKNOWN = -1


class EDaikonMagField(BaseEnum):
    """\
    [ Internal ]

    Magnetic Field
    --------------
        - OFF          : Magnetic field off.
        - NORMAL       : Normal magnetic field.
        - REVERSED     : Reversed magnetic field.
        - NEW_NORMAL   : New normal magnetic field.
        - NEW_REVERSED : New reversed magnetic field.
        - UNKNOWN      : Unknown magnetic field configuration.
    """

    OFF = 0
    NORMAL = 1
    REVERSED = 2
    NEW_NORMAL = 3
    NEW_REVERSED = 4
    UNKNOWN = -1

    @classmethod
    def _missing_(cls, value: object) -> EDaikonMagField:
        return cls.UNKNOWN
