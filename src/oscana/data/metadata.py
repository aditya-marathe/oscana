"""\
oscana / data / metadata.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = ["FileMetadataEnum", "DetectorEnum", "SimFlagEnum", "FileMetadata"]

from typing import Literal

from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# ============================== [ Constants  ] ============================== #


class DetectorEnum(Enum):
    Near = 1
    Far = 2
    Unknown = -1

    @classmethod
    def _missing_(cls, value: object) -> DetectorEnum:
        return cls(cls.Unknown)

    def __str__(self) -> str:
        return self.name


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


class FileMetadataEnum(Enum):
    # Detectors
    Far = "Far"
    Near = "Near"
    # Neutrino Source
    Beam = "Beam"
    Atmospheric = "Atmospheric"
    # Unknown
    Unknown = "Unknown"

    def __str__(self) -> str:
        return self.name


_sumamry_text = """\
{0!s}
{1!s}
Experiment      : {2!s}
Detector        : {3!s}
File Type       : {4!s}
Neutrino Source : {5!s}
Version         : {6!s}
Total Entries   : {7:,}
Run Number      : {10:,}
Date and Time
    Start       : {8!s}
    End         : {9!s}
First Loaded On : {11!s}
"""


# ============================ [ File Metadata  ] ============================ #


@dataclass(frozen=True)
class FileMetadata:
    """\
    [ Internal ] 
    
    Dataclass for storing metadata of SNTP files.
    
    Parameters
    ----------
    file_name : str
        The name of the file.

    experiment : Literal["MINOS", "NOvA", "DUNE"]
        The experiment the file is from.

    detector : Literal["Near", "Far"]
        The detector: either "near" or "far".

    file_type : Literal["MC", "Data"]
        The type of the file: either MC or (experiment) data.

    nu_source : Literal["Beam", "Atmospheric"]
        The source of the neutrinos: either "beam" or "atmpspheric".

    version : str
        Analysis or MC version e.g. "Dogwood 7".
    
    start_datetime : datetime
        The start date and time of the data collection.

    end_datetime : datetime
        The end date and time of the data collection.

    run_number : int
        The run number of the data collection.   
    """

    # Basic Information
    file_name: str
    experiment: Literal["MINOS"]  # Currently only MINOS is supported.
    detector: DetectorEnum

    # Contents
    file_type: SimFlagEnum
    nu_source: Literal[
        FileMetadataEnum.Beam,
        FileMetadataEnum.Atmospheric,
        FileMetadataEnum.Unknown,
    ]
    version: str
    n_records: int

    # Date and Time
    start_datetime: datetime
    end_datetime: datetime

    # Run Information
    run_number: int

    # More book-keeping
    creation_datetime: datetime = datetime.now()

    def print(self) -> None:
        title = self.file_name + "'s Metadata"
        print(
            _sumamry_text.format(
                title,
                "-" * len(title),
                self.experiment,
                self.detector,
                self.file_type,
                self.nu_source,
                self.version,
                self.n_records,
                self.start_datetime.strftime("%d-%m-%Y %H:%M:%S"),
                self.end_datetime.strftime("%d-%m-%Y %H:%M:%S"),
                self.run_number,
                self.creation_datetime.strftime("%d-%m-%Y %H:%M:%S"),
            )
        )

    def __str__(self) -> str:
        return f"Oscana.{self.__class__.__name__}(file='{self.file_name}')"

    def __repr__(self) -> str:
        return str(self)
