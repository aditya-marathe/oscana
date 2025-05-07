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
    "create_fd_full_image",
    "create_fd_split_image",
    "image_to_sparse",
]

import logging

import numpy as np
import numpy.typing as npt
import scipy.sparse as sps

from .logger import _error
from .utils import minos_numbers
from .constants import IMAGE_DTYPE, EPlaneView

# ================================ [ Logger ] ================================ #

_logger = logging.getLogger("Root")

# =========================== [ Helper Functions ] =========================== #


def image_to_sparse(image: npt.NDArray[IMAGE_DTYPE]) -> sps.csr_matrix:
    """\
    Convert a dense image to a sparse matrix.

    Parameters
    ----------
    image : npt.NDArray[IMAGE_DTYPE]
        The dense image to convert.

    Returns
    -------
    sps.csr_matrix
        The sparse matrix representation of the image.
    """
    return sps.csr_matrix(image, shape=image.shape, dtype=IMAGE_DTYPE)


# ============================== [ Functions  ] ============================== #


def create_fd_full_image(
    plane: EPlaneView,
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    fill: list[npt.NDArray] | None = None,
) -> npt.NDArray[IMAGE_DTYPE]:
    """\
    Get the FD event image for the given plane.

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

    fill : list[npt.NDArray] | None
        Array(s) to fill the image. Defaults to `None`. If `None`, the image
        will be filled with "1"s.

    Returns
    -------
    npt.NDArray[IMAGE_DTYPE]
        The FD event image for the given plane.
    """
    # (1) Run checks on the user input.

    # TODO: Do these checks slow this code down significantly?
    # stp_planeview = np.asarray(stp_planeview, dtype=np.uint16)
    # stp_strip = np.asarray(stp_strip, dtype=np.uint16)
    # stp_plane = np.asarray(stp_plane, dtype=np.uint16)

    if not (stp_planeview.shape == stp_strip.shape == stp_plane.shape):
        _error(
            ValueError,
            "The `stp.planeview`, `stp.strip` and `stp.plane` arrays should "
            "have the same shape!",
            _logger,
        )

    if fill is None:
        fill = [np.ones(shape=stp_planeview.shape, dtype=IMAGE_DTYPE)]

    fd_s_n_planes = minos_numbers["FD"]["South"]["NPlanes"]
    fd_n_n_planes = minos_numbers["FD"]["North"]["NPlanes"]
    fd_n_planes = fd_s_n_planes + fd_n_n_planes
    fd_n_strips = minos_numbers["FD"]["NStripsPerPlane"]

    # (2) Get data for the selected plane.

    try:
        plane_selector = stp_planeview == plane.value
    except ValueError:
        _error(
            ValueError,
            "The `stp_planeview` array should be a 1D array!",
            _logger,
        )

    # Note: 'stp.plane' is 1-indexed, while 'stp.strip' is 0-indexed!

    stp_strip = stp_strip[plane_selector]
    stp_plane = stp_plane[plane_selector] - np.array(1, dtype=stp_plane.dtype)

    # (3) Fill the image.

    # Note: Shape of the image should be (HEIGHT, WIDTH, CHANNELS).

    image = np.zeros(
        shape=(fd_n_strips, fd_n_planes, len(fill)), dtype=IMAGE_DTYPE
    )

    for i, fill_value in enumerate(fill):
        # (3.1) Run checks on the fill value.

        # TODO: Do these checks slow this code down significantly?
        # fill_value = np.asarray(fill_value, dtype=IMAGE_DTYPE)

        fill_value = fill_value[plane_selector]

        if fill_value.shape != stp_strip.shape:
            _error(
                ValueError,
                f"The `fill` array #{i + 1} should have the same shape as "
                "'stp.strip', 'stp.plane' and 'stp.planeview'!",
                _logger,
            )

        # (3.2) Fill the image.

        image[stp_strip, stp_plane, i] = fill_value

    return image


def create_fd_split_image(
    plane: EPlaneView,
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    fill: list[npt.NDArray] | None = None,
) -> tuple[npt.NDArray[IMAGE_DTYPE], npt.NDArray[IMAGE_DTYPE]]:
    """\
    Get the FD event image, split into the South and North submodules, for the 
    given plane.

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

    fill : list[npt.NDArray] | None
        Array(s) to fill the image. Defaults to `None`. If `None`, the image
        will be filled with "1"s.

    Returns
    -------
    tuple[npt.NDArray[IMAGE_DTYPE], npt.NDArray[IMAGE_DTYPE]]
        The FD event image for the given plane, split into the South and North
        submodules.
    """
    # (1) Get the full image.

    full_image = create_fd_full_image(
        plane=plane,
        stp_planeview=stp_planeview,
        stp_strip=stp_strip,
        stp_plane=stp_plane,
        fill=fill,
    )

    # (2) Split the image into the South and North submodules.

    fd_s_n_planes = minos_numbers["FD"]["South"]["NPlanes"]

    return full_image[:, :fd_s_n_planes, :], full_image[:, fd_s_n_planes:, :]
