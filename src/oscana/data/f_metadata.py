"""\
oscana / data / f_metadata.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = []

from typing import Any

# Note: There is a compatibility issue with `dataclasses`--in older versions of
#       Python, the `slots` parameter is not available. I will solve this issue
#       if it arises.

import logging, re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .enumerations import (
    EFileFormat,
    EDetector,
    EExperiment,
    # ESimFlag,  # --> Not used for now...
    EMCVersion,
    ERecoVersion,
    EDaikonIntRegion,
    EDaikonFlavour,
    EDaikonMagField,
    EHornPosition,
    EHornCurrent,
    EFileType,
)
from ..utils import (
    OscanaError,
    _error,
    _convert_from_utc,
    SNTP_BR_STD,
    SNTP_VR_RUN,
    SNTP_VR_EVT_UTC,
)

# ================================ [ Logger ] ================================ #

logger = logging.getLogger("Root")

# ============================== [ Constants  ] ============================== #

_sumamry_text = """\
{0!s}
{1!s}
File Format     : {2!s}
File Type       : {3!s}
Experiment      : {4!s}
Detector        : {5!s}
Int. Region     : {6!s}
Flavour         : {7!s}
Mag. Field      : {8!s}
Horn Position   : {9!s}
Target Z Shift  : {10!s} cm
Curr. Direction : {11!s}
Current         : {12!s} kAmps
Run Number      : {13:,}
MC Version      : {14!s} {15!s}
Reco. Version   : {16!s} {17!s}
Date and Time
    Start       : {18!s}
    End         : {19!s}
Total Entries   : {20:,}
First Loaded On : {21!s}
"""

_daikon_key_map: dict[str, tuple[EDetector, EFileType]] = {
    "n1": (EDetector.NEAR, EFileType.MONTE_CARLO),
    "f2": (EDetector.FAR, EFileType.MONTE_CARLO),
    "F2": (EDetector.FAR, EFileType.UNKNOWN),  # Mock Data?
}

_daikon_regex = (
    r"^(?P<det>n1|f2|F2)"  # Detector
    r"(?P<int>[012345])"  # Interaction
    r"(?P<flr>[012349])"  # Flavour
    r"(?P<fld>[01234])"  # BField
    r"(?P<run>[1-9]\d{3})_"  # Run Number (MC)
    r"(?P<sun>\d{4})_"  # Sub Run Number (MC)
    r"(?P<pos>[LMH])"  # Horn Position
    r"(?P<zst>000|010|100|150|250)"  # Target Z Shift
    r"(?P<cur>000|170|185|200)"  # Current
    r"(?P<sgn>[NR])_"  # Current Sign
    r"(?P<veg>[ABCD])"  # MC Vegetable
    r"(?P<ver>0[1-9]|1[0-9])_"  # MC Vegetable Version
    r"(?:r)(?P<bfr>\d+)\."  # Run Number (Beam Flux)
    r"(?P<flt>sntp|cand|mrnt)\."  # File Type
    r"(?P<wod>ash|birch|dogwood)"  # Release Wood
    r"(?P<wer>\d+\.\d+)"  # Release Wood Version
)

# =========================== [ Helper Functions ] =========================== #


def _get_sntp_run_number(file_name: str, ntpst_branch: Any) -> int:
    """\
    [ Internal ]

    Get the run number from the file.

    Notes
    -----
    Called by `_get_sntp_metadata`.
    """
    run_numbers = ntpst_branch[SNTP_VR_RUN].arrays(library="np")[
        SNTP_VR_RUN.split("/")[-1]
    ]

    if np.all(run_numbers == run_numbers[0]):
        return run_numbers[0]

    logger.warning(f"Multiple run numbers found in the file '{file_name}'!")

    # Note: Here I am finding the mode of the run number just in case the file
    #       has more than one run stored in it (which is unlikely).
    values, counts = np.unique(run_numbers, return_counts=True)
    run_number = values[np.argmax(counts)]

    return run_number


def _get_sntp_datetime_and_len(
    ntpst_branch: Any,
) -> tuple[datetime, datetime, int]:
    """\
    [ Internal ]

    Get the start and end date and time from the file.

    Notes
    -----
    Called by `_get_sntp_metadata`.
    """
    time_array = _convert_from_utc(
        utc_timestamps=ntpst_branch[SNTP_VR_EVT_UTC].arrays(library="np")[
            SNTP_VR_EVT_UTC.split("/")[-1]
        ]
    )

    # Note: A simple `type` will tell you that np.min(...) and np.max(...)
    #       return a `datetime.datetime` object not a NumPy `datetime64` object!
    #       So I have decided to ignore the `reportReturnType` warning here.
    #
    #       Weird "typing" going on here...

    return (  # pyright: ignore[reportReturnType]
        np.min(time_array),
        np.max(time_array),
        len(time_array),
    )


def _parse_file_name_spill_daikon(file_name: str) -> dict[str, Any] | None:
    """\
    [ Internal ]
    
    Parse the file name of a Daikon version MC spill file.
    
    Parameters
    ----------
    file_name : str
        Name of the file (i.e. something.extension).
        
    Returns
    -------
    None | dict[str, Any]"""
    result = re.match(_daikon_regex, file_name)
    file_format = EFileFormat(file_name.split(".")[-1])

    if (result is None) or (file_format == EFileFormat.UNKNOWN):
        return

    detector, file_type = _daikon_key_map[result.group("det")]

    return {
        "file_name": file_name,
        "file_format": file_format,
        "file_type": file_type,
        "experiment": EExperiment.MINOS,
        "detector": detector,
        "interaction": EDaikonIntRegion(int(result.group("int"))),
        "flavour": EDaikonFlavour(int(result.group("flr"))),
        "mag_field": EDaikonMagField(int(result.group("fld"))),
        "horn_pos": EHornPosition(result.group("pos")),
        "tgt_z_shift": int(result.group("zst")),
        "current_sign": EHornCurrent(result.group("sgn")),
        "current": int(result.group("cur")),
        "run_number": int(result.group("bfr")),
        "mc_version": (
            EMCVersion(result.group("veg")),
            int(result.group("ver")),
        ),
        "reco_version": (
            ERecoVersion(result.group("wod")),
            float(result.group("wer")),
        ),
    }


def from_daikon_sntp(file_name: str, file: Any) -> FileMetadata | None:
    """\
    [ Internal ] 
    
    Create a new instance of `FileMetadata` from a Daikon version MC SNTP file.
    
    Parameters
    ----------
    file_name : str
        Name of the file.
    file : Any
        The file object.
    
    Returns
    -------
    Self
        New instance of `FileMetadata`.
    """
    # Note: When you only use the TTree name, Uproot will load the latest
    #       version of the TTree automatically!
    start_time, end_time, n_records = _get_sntp_datetime_and_len(
        ntpst_branch=file[SNTP_BR_STD]
    )

    parsed_file_name = _parse_file_name_spill_daikon(file_name=file_name)

    if parsed_file_name is None:
        return

    parsed_file_name.update(
        {
            "start_time": start_time,
            "end_time": end_time,
            "n_records": n_records,
        }
    )

    return FileMetadata(**parsed_file_name)


# ============================ [ File Metadata  ] ============================ #


@dataclass(frozen=True, slots=True)
class FileMetadata:
    """\
    [ Internal ] 
    
    Dataclass for storing metadata of MC / Data files.
    
    Parameters
    ----------
    file_name: str
        Name of the file (e.g., file.root).
    file_format: EFileFormat
        Format of the file.
    file_type: EFileType
        Type of the file (e.g. Data / MC).
    experiment: EExperiment
        Experiment name.
    detector: EDetector
        Detector name.
    interaction: EDaikonIntRegion
        Interaction region.
    flavour: EDaikonFlavour
        Flavour of the beam.
    mag_field: EDaikonMagField
        Magnetic field.
    horn_pos: EHornPosition
        Horn position.
    tgt_z_shift: int
        Target Z shift.
    current: int
        Current in the horn.
    run_number: int
        Run number.
    mc_version: tuple[EMCVersion, int]
        MC version (Vegetable) and version number.
    reco_version: tuple[ERecoVersion, int]
        Reconstruction version (Tree) and version number.
    start_time: datetime
        Start time in the file.
    end_time: datetime
        End time in the file.
    n_records: int
        Total number of records in the file.
    create_time: datetime
        Time when the metadata was created. Defaults to the current time.
    """

    file_name: str
    file_format: EFileFormat
    file_type: EFileType

    experiment: EExperiment
    detector: EDetector
    interaction: EDaikonIntRegion
    flavour: EDaikonFlavour
    mag_field: EDaikonMagField
    horn_pos: EHornPosition
    tgt_z_shift: int
    current_sign: EHornCurrent
    current: int

    run_number: int

    mc_version: tuple[EMCVersion, int]
    reco_version: tuple[ERecoVersion, float]

    start_time: datetime
    end_time: datetime
    n_records: int

    create_time: datetime = datetime.now()

    @staticmethod
    def from_sntp(file_name: str, file: Any) -> FileMetadata:
        daikon = from_daikon_sntp(file_name=file_name, file=file)
        if daikon is not None:
            return daikon

        _error(
            OscanaError,
            (
                f"File '{file_name}' does not match any known SNTP format! "
                "Failed to extract file metadata."
            ),
            logger,
        )

    def print(self) -> None:
        print(
            _sumamry_text.format(
                self.file_name,
                "-" * len(self.file_name),
                self.file_format,
                self.file_type,
                self.experiment,
                self.detector,
                self.interaction,
                self.flavour,
                self.mag_field,
                self.horn_pos,
                self.tgt_z_shift,
                self.current_sign,
                self.current,
                self.run_number,
                self.mc_version[0],
                self.mc_version[1],
                self.reco_version[0],
                self.reco_version[1],
                self.start_time,
                self.end_time,
                self.n_records,
                str(self.create_time)[:19],
            )
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, FileMetadata):
            _error(
                ValueError,
                (
                    f"{self.__class__.__name__} can only be compared with "
                    "another instance of the same class."
                ),
                logger,
            )

        return (
            (self.file_type == value.file_type)
            and (self.experiment == value.experiment)
            and (self.detector == value.detector)
            # Allow different interaction regions in the same file.
            # and (self.interaction == value.interaction)
            and (self.flavour == value.flavour)
            and (self.mag_field == value.mag_field)
            and (self.horn_pos == value.horn_pos)
            and (self.tgt_z_shift == value.tgt_z_shift)
            and (self.current_sign == value.current_sign)
            and (self.current == value.current)
            and (self.run_number == value.run_number)
            and (self.mc_version[0] == value.mc_version[0])
            and (self.mc_version[1] == value.mc_version[1])
            and (self.reco_version[0] == value.reco_version[0])
            and (self.reco_version[1] == value.reco_version[1])
        )

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __str__(self) -> str:
        return f"Oscana.{self.__class__.__name__}(file='{self.file_name}')"

    def __repr__(self) -> str:
        return str(self)
