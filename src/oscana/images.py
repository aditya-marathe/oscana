"""\
oscana / images.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

This module contains functions to extract the event images from the MINOS SNTP 
files. Only works for data from MINOS!
"""

from __future__ import annotations

from typing import List
from typing import Optional
from typing import TypedDict

__all__ = [
    "image_to_sparse",
    "get_image_profiles",
    "create_fd_full_image",
    "create_fd_split_image",
    "create_fd_crop_image",
]

import logging

import numpy as np
import numpy.typing as npt
import scipy.sparse as sps

from .logger import _error, _warn
from .utils import minos_numbers, get_bin_centers
from .constants import IMAGE_DTYPE, EPlaneView

# ================================ [ Logger ] ================================ #

_logger = logging.getLogger("Root")

# ============================== [ Constants  ] ============================== #


class StripPlaneProfile(TypedDict):
    """\
    A profile for either a strip or a plane.
    """

    BinHeights: npt.NDArray
    BinEdges: npt.NDArray
    BinCenters: npt.NDArray
    Mean: float
    StD: float
    Min: float
    Max: float
    Median: float


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

    Notes
    -----
    More specifically, this function converts the input array into a Compressed
    Sparse Row (CSR) matrix format.
    """
    return sps.csr_matrix(image, shape=image.shape, dtype=IMAGE_DTYPE)


def get_image_profiles(
    plane: EPlaneView,
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    weight_variable: Optional[npt.NDArray] = None,
) -> tuple[StripPlaneProfile, StripPlaneProfile]:
    """\
    Get weighted strip and plane profiles for the given plane.

    Parameters
    ----------
    plane : EPlaneView
        The plane view to extract the profiles for (either U-Z or V-Z).

    stp_planeview : npt.NDArray
        The `stp.planeview` variable from the SNTP_BR_STD branch of SNTP files.

    stp_strip : npt.NDArray
        The `stp.strip` variable from the SNTP_BR_STD branch of SNTP files.
    
    stp_plane : npt.NDArray
        The `stp.plane` variable from the SNTP_BR_STD branch of SNTP files.
    
    weight_variable : Optional[npt.NDArray]
        The weights to apply to the profiles. Defaults to `None`.

    Returns
    -------
    tuple[StripPlaneProfile, StripPlaneProfile]
        The strip and plane profiles for the given plane, respectively.

    Notes
    -----
    Keys in the output dictionaries:
        - "BinHeights": The heights of the histogram bins.
        - "BinEdges": The edges of the histogram bins.
        - "BinCenters": The centers of the histogram bins.
        - "Mean": The mean of the variable.
        - "StD": The standard deviation of the variable.
        - "Min": The minimum value of the variable.
        - "Max": The maximum value of the variable.
        - "Median": The median value of the variable.
    """
    # (1) Preparing the variables.
    fd_n_planes = (
        minos_numbers["FD"]["South"]["NPlanes"]
        + minos_numbers["FD"]["North"]["NPlanes"]
    )
    fd_n_strips = minos_numbers["FD"]["NStripsPerPlane"]

    plane_selector = stp_planeview == plane.value
    stp_strip = stp_strip[plane_selector]
    stp_plane = stp_plane[plane_selector] - np.array(1, dtype=stp_plane.dtype)

    if weight_variable is not None:
        weight_variable = weight_variable[plane_selector]

    # (2) Strip profile.
    strip_bin_heights, strip_bin_edges = np.histogram(
        stp_strip, bins=np.arange(0, fd_n_strips, 1), weights=weight_variable
    )

    std_dev = float(np.std(stp_strip))

    strip_profile: StripPlaneProfile = {
        # Histogram
        "BinHeights": np.asarray(strip_bin_heights, dtype=float),
        "BinEdges": np.asarray(strip_bin_edges, dtype=float),
        "BinCenters": get_bin_centers(bin_edges=strip_bin_edges),
        # Stats
        "Mean": float(np.mean(stp_strip)),
        "StD": std_dev if (std_dev > 0) else 1.0,
        "Min": float(np.min(stp_strip)),
        "Max": float(np.max(stp_strip)),
        "Median": float(np.median(stp_strip)),
    }

    # (3) Plane profile.
    plane_bin_heights, plane_bin_edges = np.histogram(
        stp_plane,
        bins=np.arange(1, fd_n_planes + 1, 1),
        weights=weight_variable,
    )

    std_dev = float(np.std(stp_plane))

    plane_profile: StripPlaneProfile = {
        # Histogram
        "BinHeights": np.asarray(plane_bin_heights, dtype=float),
        "BinEdges": np.asarray(plane_bin_edges, dtype=float),
        "BinCenters": get_bin_centers(bin_edges=plane_bin_edges),
        # Stats
        "Mean": float(np.mean(stp_plane)),
        "StD": std_dev if (std_dev > 0) else 1.0,
        "Min": float(np.min(stp_plane)),
        "Max": float(np.max(stp_plane)),
        "Median": float(np.median(stp_plane)),
    }

    return strip_profile, plane_profile


def _crop_image(
    image: npt.NDArray, x: int, y: int, width: int, height: int
) -> npt.NDArray:
    """\
    [ Internal ] Crop an image to the given dimensions.
    
    Parameters
    ----------
    image : npt.NDArray
        The image to crop.
        
    x : int
        The x-coordinate of the top-left corner of the crop.
        
    y : int
        The y-coordinate of the top-left corner of the crop.
        
    width : int
        The width of the crop.
        
    height : int
        The height of the crop.
    
    Returns
    -------
    npt.NDArray
        The cropped image.
    """
    image_height, image_width = image.shape[:2]
    image_channels_tuple = image.shape[2:]

    # (1) Initialise the output image array.
    cropped_image = np.zeros(
        shape=(height, width, *image_channels_tuple), dtype=image.dtype
    )

    if (x > (image_width - 1)) or ((y - height) > image_height) or (y < 0):
        _warn(
            RuntimeWarning,
            f"The crop coordinate ({x}, {y}) is outside the image with width "
            f"{image_width} and height {image_height}! Returned a blank image.",
            _logger,
        )
        return cropped_image  # This should not happen!.

    # (2) Calculate the cropping indices. Why is this such a headache?

    # (2.1) These are the full image indices.

    img_x_beg = max(0, x)
    img_x_end = min(image_width, x + width)

    img_y_beg = max(0, y - height)
    img_y_end = min(image_height, y)

    # (2.2) These are the cropped image indices.

    out_x_beg = max(0, -x)  # If, for some reason, x < 0
    out_x_end = out_x_beg + (img_x_end - img_x_beg)

    out_y_beg = img_y_beg - (y - height)
    out_y_end = out_y_beg + (img_y_end - img_y_beg)

    cropped_image[out_y_beg:out_y_end, out_x_beg:out_x_end, ...] = image[
        img_y_beg:img_y_end, img_x_beg:img_x_end, ...
    ]  # ~ note that the `...` is there for a reason! - do not change this

    return cropped_image


# ============================== [ Functions  ] ============================== #


def create_fd_full_image(
    plane: EPlaneView,
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    fill: Optional[List[npt.NDArray]] = None,
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

    fill : Optional[List[npt.NDArray]]
        Array(s) to fill the image. Defaults to `None`. If `None`, the image
        will be filled with "1"s.

    Returns
    -------
    npt.NDArray[IMAGE_DTYPE]
        The FD event image for the given plane.
    """
    # (1) Run checks on the user input.

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
    fill: Optional[List[npt.NDArray]] = None,
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

    fill : Optional[List[npt.NDArray]]
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


def create_fd_crop_image(
    plane: EPlaneView,
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    stp_pe_east: npt.NDArray,
    stp_pe_west: npt.NDArray,
    cropped_width: int,
    cropped_height: int,
    fill: Optional[List[npt.NDArray]] = None,
    constrain_strip: Optional[float] = None,
    constrain_plane: Optional[float] = None,
) -> npt.NDArray[IMAGE_DTYPE]:
    """\
    Create a cropped image from the full FD image for a particular plane.

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

    stp_pe_east : npt.NDArray
        The pulse height (photoelectrons) for the east side.

    stp_pe_west : npt.NDArray
        The pulse height (photoelectrons) for the west side.

    cropped_width: int
        The width of the cropped image.

    cropped_height: int
        The height of the cropped image.

    fill : Optional[List[npt.NDArray]]
        Array(s) to fill the image. Defaults to `None`. If `None`, the image
        will be filled with "1"s.

    constrain_strip : Optional[float]
        Only keep `constrain_strip` number of sigmas from the strip center of
        mass. If `None`, no constraint is applied. Defaults to `None`.

    constrain_plane : Optional[float]
        Only keep `constrain_plane` number of sigmas from the plane center of
        mass. If `None`, no constraint is applied. Defaults to `None`.

    Returns
    -------
    npt.NDArray[IMAGE_DTYPE]
        The cropped FD event image for the selected plane.        
    """
    # (1) Get the mean PE-weighted plane and strip profiles.
    strip_profile, plane_profile = get_image_profiles(
        plane=plane,
        stp_planeview=stp_planeview,
        stp_strip=stp_strip,
        stp_plane=stp_plane,
        weight_variable=(stp_pe_east + stp_pe_west) / 2,
    )

    # (2) Calculate the crop co-ordinates etc.
    image_constrain = np.ones_like(stp_planeview).astype(bool)

    if constrain_strip:
        image_constrain &= np.abs(stp_strip - strip_profile["Mean"]) < (
            constrain_strip * strip_profile["StD"]
        )

    if constrain_plane:
        image_constrain &= np.abs(stp_plane - plane_profile["Mean"]) < (
            constrain_plane * plane_profile["StD"]
        )

    # (3) Get the full image.
    if constrain_strip or constrain_plane:
        fill = (
            [f[image_constrain] for f in fill] if (fill is not None) else None
        )

    image = create_fd_full_image(
        plane=plane,
        stp_planeview=stp_planeview[image_constrain],
        stp_strip=stp_strip[image_constrain],
        stp_plane=stp_plane[image_constrain],
        fill=fill,
    )

    # (4) Create the cropped image.

    # TODO: This is being re-calculated three times in this function call!
    plane_selector = stp_planeview == plane.value

    return _crop_image(
        image=image,
        x=(int(stp_plane[image_constrain & plane_selector].min()) - 1),
        y=(int(strip_profile["Mean"]) + cropped_height // 2),
        width=cropped_width,
        height=cropped_height,
    )
