"""\
oscana / evd / image.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = ["IMAGE_VARIABLES", "get_fd_pixel_images"]

import logging

import numpy as np
import numpy.typing as npt

from ..utils import SNTP_BR_STD, PlaneView, minos_numbers, _error

# ================================ [ Logger ] ================================ #

logger = logging.getLogger("Root")

# ========================== [ Constants / Enums  ] ========================== #

IMAGE_VARIABLES = [
    f"{SNTP_BR_STD}/stp.planeview",
    f"{SNTP_BR_STD}/stp.strip",
    f"{SNTP_BR_STD}/stp.plane",
    f"{SNTP_BR_STD}/stp.ph0.pe",
    f"{SNTP_BR_STD}/stp.ph1.pe",
]

# ============================== [ Functions  ] ============================== #


def get_fd_pixel_images(
    plane: PlaneView,
    stp_planeview: npt.ArrayLike,
    stp_strip: npt.ArrayLike,
    stp_plane: npt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    # Ensure NumPy arrays
    stp_planeview = np.asarray(stp_planeview)
    stp_strip = np.asarray(stp_strip)
    stp_plane = np.asarray(stp_plane)

    # Select the relevant plane (i.e. U or V)
    try:
        plane_selection = stp_planeview == plane.value
    except ValueError:
        # Note: This raises an error for now because doing something like
        #
        #       `list(map(lambda x: x == 2, stp_planeview))`
        #
        #       just seems silly (for now).

        _error(
            ValueError,
            "The `stp_planeview` array should be a 1D array.",
            logger,
        )

    stp_strip = stp_strip[plane_selection]
    stp_plane = stp_plane[plane_selection] - 1

    # Useful variables from the MINOS numbers
    fd_w_n_planes = minos_numbers["FD"]["West"]["NPlanes"]
    fd_e_n_planes = minos_numbers["FD"]["East"]["NPlanes"]
    fd_n_strips = minos_numbers["FD"]["NStripsPerPlane"]

    # Seperate the West and East modules

    # Note: 'stp.plane' is 1-indexed, while 'stp.strip' is 0-indexed! So, for
    #       West the plane number ranges from 0 to 241 and for East it ranges
    #       from 242 to 484.

    #                      0            241
    strip_west = stp_strip[stp_plane <= fd_w_n_planes - 1]
    plane_west = stp_plane[stp_plane <= fd_w_n_planes - 1]

    strip_east = stp_strip[stp_plane > fd_w_n_planes - 1]
    plane_east = stp_plane[stp_plane > fd_w_n_planes - 1] - fd_w_n_planes

    # Fill the matrices
    image_west = np.zeros(shape=(fd_n_strips, fd_w_n_planes))
    image_east = np.zeros(shape=(fd_n_strips, fd_e_n_planes))

    image_west[strip_west, plane_west] = 1.0
    image_east[strip_east, plane_east] = 1.0

    return image_west, image_east
