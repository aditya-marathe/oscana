"""\
oscana / images.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

This module contains functions to extract the event images from the SNTP files.
"""

from __future__ import annotations

__all__ = [
    "get_fd_event_images",
    "get_sparase_fd_event_images",
]

import logging

import numpy as np
import numpy.typing as npt
import scipy.sparse as sps

from .utils import minos_numbers, _error
from .constants import IMAGE_DTYPE, EPlaneView

# ================================ [ Logger ] ================================ #

_logger = logging.getLogger("Root")

# =========================== [ Helper Functions ] =========================== #


def _get_strip_plane_indices(
    plane: EPlaneView,
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    stp_ph0_pe: npt.NDArray | None,
    stp_ph1_pe: npt.NDArray | None,
) -> tuple[
    tuple[npt.NDArray, npt.NDArray],
    tuple[npt.NDArray, npt.NDArray],
    tuple[npt.NDArray, npt.NDArray],
]:
    """\
    [ Internal ] Get the strip and plane indices for the given plane view.

    Parameters
    ----------
    plane : EPlaneView
        The plane view to extract the images for (either U-Z or V-Z).

    stp_planeview : npt.NDArray
        The `stp.planeview` variable from the SNTP_BR_STD branch of SNTP files.

    stp_strip : npt.NDArray
        The `stp.strip` variable from the SNTP_BR_STD branch of SNTP files.

    stp_plane : npt.NDArray
        The `stp.plane` variable from the SNTP_BR_STD branch of SNTP files.

    stp_ph0_pe : npt.NDArray | None
        The `stp.ph0.pe` variable from the SNTP_BR_STD branch of SNTP files.
    
    stp_ph1_pe : npt.NDArray | None
        The `stp.ph1.pe` variable from the SNTP_BR_STD branch of SNTP files.

    Returns
    -------
    tuple[
        tuple[npt.NDArray, npt.NDArray],
        tuple[npt.NDArray, npt.NDArray],
        tuple[npt.NDArray, npt.NDArray],
    ]
        The strip and plane indices for the west and east modules. The PE values
        for the west and east modules.
    """
    stp_planeview = np.asarray(stp_planeview)
    stp_strip = np.asarray(stp_strip)
    stp_plane = np.asarray(stp_plane)

    try:
        plane_selection = stp_planeview == plane.value
    except ValueError:
        _error(
            ValueError,
            "The `stp_planeview` array should be a 1D array.",
            _logger,
        )

    stp_strip = stp_strip[plane_selection]
    stp_plane = stp_plane[plane_selection] - np.int64(1)

    fd_w_n_planes = minos_numbers["FD"]["West"]["NPlanes"]

    # Seperate the West and East modules

    # Note: 'stp.plane' is 1-indexed, while 'stp.strip' is 0-indexed! So, for
    #       West the plane number ranges from 0 to 241 and for East it ranges
    #       from 242 to 484.

    west_selection = stp_plane <= (fd_w_n_planes - 1)
    east_selection = stp_plane > (fd_w_n_planes - 1)

    strip_west = stp_strip[west_selection]
    plane_west = stp_plane[west_selection]

    strip_east = stp_strip[east_selection]
    plane_east = stp_plane[east_selection] - fd_w_n_planes

    if (stp_ph0_pe is not None) and (stp_ph1_pe is not None):
        stp_ph1_pe = stp_ph1_pe[plane_selection][west_selection]
        stp_ph0_pe = stp_ph0_pe[plane_selection][east_selection]
    else:
        stp_ph1_pe = np.ones_like(strip_west, dtype=IMAGE_DTYPE)
        stp_ph0_pe = np.ones_like(strip_east, dtype=IMAGE_DTYPE)

    return (
        (strip_west, plane_west),
        (strip_east, plane_east),
        (stp_ph1_pe, stp_ph0_pe),
    )


# ============================== [ Functions  ] ============================== #


def get_fd_event_images(
    plane: EPlaneView,
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    stp_ph0_pe: npt.NDArray | None = None,
    stp_ph1_pe: npt.NDArray | None = None,
) -> tuple[npt.NDArray[IMAGE_DTYPE], npt.NDArray[IMAGE_DTYPE]]:
    """\
    Get the west and east FD event images for the given plane.

    Parameters
    ----------
    plane : EPlaneView
        The plane view to extract the images for (either U-Z or V-Z).

    stp_planeview : npt.NDArray
        The `stp.planeview` variable from the SNTP_BR_STD branch of SNTP files.

    stp_strip : npt.NDArray
        The `stp.strip` variable from the SNTP_BR_STD branch of SNTP files.

    stp_plane : npt.NDArray
        The `stp.plane` variable from the SNTP_BR_STD branch of SNTP files.

    stp_ph0_pe : npt.NDArray | None
        The `stp.ph0.pe` variable from the SNTP_BR_STD branch of SNTP files.
        Defaults to `None`.

    stp_ph1_pe : npt.NDArray | None
        The `stp.ph1.pe` variable from the SNTP_BR_STD branch of SNTP files.
        Defaults to `None`.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        The west and east FD event images.
    """
    fd_w_n_planes = minos_numbers["FD"]["West"]["NPlanes"]
    fd_e_n_planes = minos_numbers["FD"]["East"]["NPlanes"]
    fd_n_strips = minos_numbers["FD"]["NStripsPerPlane"]

    west_indices, east_indices, digit_values = _get_strip_plane_indices(
        plane=plane,
        stp_planeview=stp_planeview,
        stp_strip=stp_strip,
        stp_plane=stp_plane,
        stp_ph0_pe=stp_ph0_pe,
        stp_ph1_pe=stp_ph1_pe,
    )

    # Fill the matrices
    image_west = np.zeros(shape=(fd_n_strips, fd_w_n_planes), dtype=IMAGE_DTYPE)
    image_east = np.zeros(shape=(fd_n_strips, fd_e_n_planes), dtype=IMAGE_DTYPE)

    image_west[west_indices] = digit_values[0]
    image_east[east_indices] = digit_values[1]

    return image_west, image_east


def get_sparase_fd_event_images(
    plane: EPlaneView,
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    stp_ph0_pe: npt.NDArray | None = None,
    stp_ph1_pe: npt.NDArray | None = None,
) -> tuple[sps.csr_matrix, sps.csr_matrix]:
    """\
    Get the west and east FD event sparse images for the given plane.

    Parameters
    ----------
    plane : EPlaneView
        The plane view to extract the images for (either U-Z or V-Z).
    
    stp_planeview : npt.NDArray
        The `stp.planeview` variable from the SNTP_BR_STD branch of SNTP files.

    stp_strip : npt.NDArray
        The `stp.strip` variable from the SNTP_BR_STD branch of SNTP files.

    stp_plane : npt.NDArray
        The `stp.plane` variable from the SNTP_BR_STD branch of SNTP files.

    stp_ph0_pe : npt.NDArray | None
        The `stp.ph0.pe` variable from the SNTP_BR_STD branch of SNTP files.
        Defaults to `None`.

    stp_ph1_pe : npt.NDArray | None
        The `stp.ph1.pe` variable from the SNTP_BR_STD branch of SNTP files.
        Defaults to `None`.

    Returns
    -------
    tuple[sps.csr_matrix, sps.csr_matrix]
        The west and east FD event sparse event images.
    """
    image_west, image_east = get_fd_event_images(
        plane=plane,
        stp_planeview=stp_planeview,
        stp_strip=stp_strip,
        stp_plane=stp_plane,
        stp_ph0_pe=stp_ph0_pe,
        stp_ph1_pe=stp_ph1_pe,
    )

    return sps.csr_matrix(image_west), sps.csr_matrix(image_east)
