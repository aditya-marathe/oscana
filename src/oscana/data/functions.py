"""\
oscana / data / functions.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = ["SNTP_BR_NTPST", "SNTP_BR_NTPBDLITE"]

from typing import Any, Literal

import logging
from datetime import datetime

import uproot
import pandas as pd
import numpy as np
import numpy.typing as npt

from ..utils import _get_dir_from_env
from .metadata import FileMetadata, FileMetadataEnum

# ================================ [ Logger ] ================================ #

logger = logging.getLogger("Root")

# ============================== [ Constants  ] ============================== #

# SNTP Branches
SNTP_BR_NTPST = "NtpSt"
SNTP_BR_NTPBDLITE = "NtpBDLite"

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
SNTP_VR_EVT_UTC = "NtpStRecord/evthdr/evthdr.date.utc"

# =========================== [ Helper Functions ] =========================== #


def _check_for_repeats(some_list: list[Any]) -> bool:
    """\
    [ Internal ]

    Check if the list has any repeated elements.
    """
    return len(some_list) != len(set(some_list))


def _get_sntp_detector(
    file_name: str, ntpst_branch: Any
) -> Literal[
    FileMetadataEnum.Near, FileMetadataEnum.Far, FileMetadataEnum.Unknown
]:
    """\
    [ Internal ]

    Get the detector from the file.

    Notes
    -----
    Called by `_get_sntp_metadata`.
    """
    detector_id = ntpst_branch[SNTP_VR_DETECTOR].arrays(library="np")[
        SNTP_VR_DETECTOR.split("/")[-1]
    ]

    if np.all(detector_id == detector_id[0]):
        if detector_id[0] == 1:
            detector = FileMetadataEnum.Near
        elif detector_id[0] == 2:
            detector = FileMetadataEnum.Far
        else:
            detector = FileMetadataEnum.Unknown

    else:
        logger.warning(f"Multiple detectors found in the file '{file_name}'!")
        detector = FileMetadataEnum.Unknown

    return detector


def _get_sntp_data_type(
    file_name: str, ntpst_branch: Any
) -> Literal[
    FileMetadataEnum.Data, FileMetadataEnum.MC, FileMetadataEnum.Unknown
]:
    """\
    [ Internal ]

    Get the data type from the file.

    Notes
    -----
    Called by `_get_sntp_metadata`.
    """
    sim_flags = ntpst_branch[SNTP_VR_SIM].arrays(library="np")[
        SNTP_VR_SIM.split("/")[-1]
    ]

    if np.all(sim_flags == sim_flags[0]):
        if sim_flags[0] == 0:
            data_type = FileMetadataEnum.Data
        elif sim_flags[0] == 1:
            data_type = FileMetadataEnum.MC
        else:
            data_type = FileMetadataEnum.Unknown

    else:
        logger.warning(f"Multiple data types found in the file '{file_name}'!")
        data_type = FileMetadataEnum.Unknown

    return data_type


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
        run_number = run_numbers[0]
    else:
        logger.warning(f"Multiple run numbers found in the file '{file_name}'!")

        # Note: Here I am finding the mode of the run number just in case the
        #       file has more than one run stored in it (which is unlikely).
        values, counts = np.unique(run_numbers, return_counts=True)
        run_number = values[np.argmax(counts)]

    return run_number


def _get_sntp_datetime(
    file_name: str, ntpst_branch: Any
) -> tuple[datetime, datetime]:
    """\
    [ Internal ]

    Get the start and end date and time from the file.

    Notes
    -----
    Called by `_get_sntp_metadata`.
    """
    utc_time_array = ntpst_branch[SNTP_VR_EVT_UTC].arrays(library="np")[
        SNTP_VR_EVT_UTC.split("/")[-1]
    ]

    # This solution was thanks to ChatGPT - it actually works sometimes! :)
    time_array = (
        np.datetime64("1970-01-01T00:00:00")
        + utc_time_array.astype("timedelta64[s]")
    ).astype("O")

    return np.max(time_array), np.min(time_array)


def _get_sntp_metadata(file_name: str, file: Any) -> FileMetadata:
    """\
    [ Internal ]

    Get the metadata of the file.
    """
    # Note: When you only use the TTree name, Uproot will load the latest
    #       version of the TTree automatically!
    ntpst_branch = file[SNTP_BR_NTPST]

    start_datetime, end_datetime = _get_sntp_datetime(
        file_name=file_name, ntpst_branch=ntpst_branch
    )

    metadata = FileMetadata(
        file_name=file_name,
        experiment=FileMetadataEnum.MINOS,
        detector=_get_sntp_detector(
            file_name=file_name, ntpst_branch=ntpst_branch
        ),
        file_type=_get_sntp_data_type(
            file_name=file_name, ntpst_branch=ntpst_branch
        ),
        nu_source=FileMetadataEnum.Unknown,
        version="???",
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        run_number=_get_sntp_run_number(
            file_name=file_name, ntpst_branch=ntpst_branch
        ),
    )

    logger.info(f"Extracted metadata for '{file_name}': {metadata!s}.")

    return metadata


def _get_dst_metadata(file_name: str, file: Any) -> FileMetadata:
    # TODO: Implement this function.

    logger.warning("DST metadata extraction is not implemented yet!")

    return FileMetadata(
        file_name=file_name,
        experiment=FileMetadataEnum.Unknown,
        detector=FileMetadataEnum.Unknown,
        file_type=FileMetadataEnum.Unknown,
        nu_source=FileMetadataEnum.Unknown,
        version="???",
        start_datetime=datetime.now(),
        end_datetime=datetime.now(),
        run_number=0,
    )


# ============================= [ File Loaders ] ============================= #


def _v1_naive_loader(
    variables: list[str], file: str
) -> tuple[pd.DataFrame, FileMetadata]:
    """\
    [ Internal ]

    V1 File Loader. A naive loader - slow, but works.
    """

    logger.debug(f"Loading variables from '{file}' using the V1 Naive Loader.")

    file_dir = _get_dir_from_env(file=file)

    uproot_file = uproot.open(file_dir)

    logger.info(f"Opened '{file}' using Uproot.")

    # This is a really crappy way to extract the metadata...
    try:
        logger.debug("Trying to extract SNTP metadata...")
        metadata = _get_sntp_metadata(file_name=file, file=uproot_file)
    except uproot.KeyInFileError:
        logger.debug("Failed! Trying to extract DST metadata instead...")
        metadata = _get_dst_metadata(file_name=file, file=uproot_file)

    data_dict: dict[str, npt.NDArray] = {}

    for variable in variables:
        logger.debug(f"Extracting variable '{variable}' from '{file}'...")

        # Note: Not great that we need to specify a base in this way.
        # TODO: Fix this.
        base = variable.split("/")[0]
        key = "/".join(variable.split("/")[1:])

        # Note: I have added some `pyright` comments to suppress annoying
        #       warnings.

        data_dict[key] = uproot_file[base][
            key
        ].arrays(  # pyright: ignore[reportAttributeAccessIssue]
            library="np"
        )[
            key.split("/")[-1]
        ]

    uproot_file.close()  # pyright: ignore[reportAttributeAccessIssue]

    logger.info(f"Extracted variables from '{file}'.")

    return pd.DataFrame(data_dict), metadata
