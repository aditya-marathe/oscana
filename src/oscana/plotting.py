"""\
oscana / plotting.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

This module contains functions to help with plotting. Using the plotting context
(`context`) allows the user to set the theme for the plot. The module also has
templates for frequently used plots, so it is possible to create publishable
plots with only one or two lines of code.
"""

from __future__ import annotations

__all__ = [
    "MINOS_GUESSED_ENERGY_BINS",
    "plotting_context",
    # Layouts
    "grid_layout",
    "spectrum_layout",
    "fd_uv_views_layout",
    # Modifiers
    "energy_axs_scale",
    "spec_fig_cleanup",
    # Plotting
    "plot_hist",
    # "plot_hist_from_heights",
    # Templates
    "plot_energy_resolution",
    "plot_fd_event_image",
    # Helpers
    "get_bin_centers",
]

from typing import Generator, Literal, Any, TYPE_CHECKING

from contextlib import contextmanager

import logging, warnings

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.scale as scl
from matplotlib import patches

from .logger import _error
from .themes import _load_settings
from .utils import minos_numbers
from .images import create_fd_split_image
from .constants import EPlaneView

if TYPE_CHECKING:
    import numpy.typing as npt

    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

# =============================== [ Logging  ] =============================== #

_logger = logging.getLogger("Plot")

# ============================== [ Constants  ] ============================== #

# This "guessed" binning will be gone as soon as I find the actual binning...
MINOS_GUESSED_ENERGY_BINS: list[float] = [
    0,
    1.0,
    1.2666666666666666,
    1.5333333333333332,
    1.8,
    2.0666666666666664,
    2.333333333333333,
    2.6,
    2.8666666666666667,
    3.1333333333333333,
    3.4,
    3.6666666666666665,
    3.933333333333333,
    4.2,
    4.466666666666667,
    4.733333333333333,
    5.0,
    5.555555555555555,
    6.111111111111111,
    6.666666666666667,
    7.222222222222222,
    7.777777777777778,
    8.333333333333334,
    8.88888888888889,
    9.444444444444445,
    10.0,
    11.11111111111111,
    12.222222222222221,
    13.333333333333334,
    14.444444444444445,
    15.555555555555555,
    16.666666666666668,
    17.77777777777778,
    18.88888888888889,
    20,
    30,
    50,
]

DEFAULT_X_AXIS_SEGMENTS: list[tuple[float, float, float]] = [
    (00, 10, 0.60),
    (10, 20, 0.25),
    (20, 30, 0.075),
    (30, 50, 0.075),
]
DEFAULT_X_AXIS_TICKS: list[float] = [0, 5, 10, 15, 20, 30, 50]

# =========================== [ Helper Functions ] =========================== #


def _ensure_axs_tuple(axs: npt.NDArray | list[Axes] | Axes) -> tuple[Axes, ...]:
    """\
    Ensure that `axs` is a list of `Axes` objects.

    Parameters
    ----------
    axs : Axes or list[Axes] or npt.NDArray[Axes]
        The axes to ensure are in a list.

    Returns
    -------
    tuple[Axes]
        Tuple of `Axes` objects.
    """

    # Note: I am going to assume that `axs` will always be one of the three
    # expected types

    if isinstance(axs, np.ndarray):
        axs_return: tuple[Axes, ...] = tuple(axs.flatten().tolist())
    elif isinstance(axs, list):
        axs_return: tuple[Axes, ...] = tuple(axs)
    else:
        axs_return: tuple[Axes, ...] = (axs,)

    return axs_return


def _axs_fwd_transform(
    segments: list[tuple[float, float, float]], array: npt.ArrayLike
) -> np.ndarray:
    """\
    Forward transform for the custom axis.

    Parameters
    ----------
    segments : List[Tuple[float, float, float]]
        The segments of the custom axis. Each segment is a tuple of the form: 
        (x_min, x_max, % of axis).
        
    array : npt.ArrayLike
        The array to be transformed to the custom axis.

    Returns
    -------
    np.ndarray
        The array transformed to the custom axis.
    """
    array = np.asarray(array)
    transformed_axis = np.zeros_like(array)

    offset = 0

    # Note: I wrote this code a really, really long time ago. Looking it at now,
    #       I have no idea what it actually does, but it seems to be working
    #       just fine.

    for segment in segments:
        idx = (array > segment[0]) & (array <= segment[1])
        transformed_axis[idx] = offset + (
            array[idx] - np.float64(segment[0])
        ) * (segment[2] / (segment[1] - segment[0]))

        offset += segment[2]

    _logger.debug("Applied the forward transform to compress the energy axis.")

    return transformed_axis


def _axs_inv_transform(
    segments: list[tuple[float, float, float]], array: npt.ArrayLike
) -> np.ndarray:
    """\
    Inverse transform for the custom axis.

    Parameters
    ----------
    segments : List[Tuple[float, float, float]]
        The segments of the custom axis. Each segment is a tuple of the form: 
        (x_min, x_max, % of axis).
        
    array : npt.ArrayLike
        The array to be transformed back to the original axis.

    Returns
    -------
    np.ndarray
        The array transformed back to the original axis.
    """
    array = np.asarray(array)
    original_axis = np.zeros_like(array)

    offset = 0

    for segment in segments:
        idx = (array > segment[0]) & (array <= segment[1])
        original_axis[idx] = segment[0] + (array[idx] - offset) / segment[2]

        offset += segment[2]

    _logger.debug(
        "Applied the inverse transform to return the energy axis to normal."
    )

    return original_axis


# =============================== [ Context  ] =============================== #


@contextmanager
def plotting_context(theme_name: str = "slate") -> Generator[None, None, None]:
    """\
    Context manager for setting the plotting theme.

    Parameters
    ----------
    theme_name : str
        Name of the theme to use for the plot. Defaults to "slate".
    """
    settings = _load_settings(theme_name=theme_name.lower())

    # Keep the original rcParameters so we can reset them later...
    original_params = {setting: mpl.rcParams[setting] for setting in settings}

    # Change the rcPrameters to our custom settings...
    mpl.rcParams.update(settings)

    # Change warnings settings, so we don't keep getting the annoying "no
    # artists found" warnings.
    warnings.simplefilter("ignore", UserWarning)

    _logger.debug(
        f"Entering the plotting context with '{theme_name}' theme. Warning "
        "messages are temporarily supressed."
    )

    try:
        yield  # Here, we are inside the context...
    finally:
        # Overwrite the rcParameters to their original values
        mpl.rcParams.update(original_params)

        # Unfilter warnings
        warnings.simplefilter("default", UserWarning)

        _logger.debug(
            "Exiting the plotting context. Warning messages are now enabled."
        )


# =============================== [ Layouts  ] =============================== #


def grid_layout(
    n_rows: int = 1,
    n_cols: int = 1,
    share_x: bool = False,
    share_y: bool = False,
    **figure_kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    """\
    Creates a simple grid layout for the plot.

    Parameters
    ----------
    n_rows : int
        Number of rows of subplots.
    
    n_cols : int
        Number of columns of subplots.
    
    share_x : bool | str
        Whether to share the x-axis.
    
    share_y : bool | str
        Whether to share the y-axis.

    Returns
    -------
    tuple[Figure, tuple[Axes, ...]]
        Matplotlib `Figure` object and a tuple of Matplotlib `Axes` object(s).
    """
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=share_x,
        sharey=share_y,
        **figure_kwargs,
    )

    _logger.debug(f"Created a {n_rows}x{n_cols} grid layout.")

    return fig, _ensure_axs_tuple(axs=axs)


def spectrum_layout(
    show_ratio: bool = False,
    show_resolution: bool = False,
    **figure_kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    """\
    Create a custom layout for a spectrum plot.

    Parameters
    ----------
    show_ratio : bool
        Whether to show the "ratio" subplot.
    
    show_resolution : bool
        Whether to show the "resolution" subplot.

    Returns
    -------
    tuple[Figure, tuple[Axes, ...]]
        Matplotlib `Figure` object and a tuple of Matplotlib `Axes` object(s).
    """
    fig = plt.figure(**figure_kwargs)

    gs = gridspec.GridSpec(
        1 + show_ratio,
        1 + show_resolution,
        figure=fig,
        height_ratios=[2.5, 1] if show_ratio else [1],
        width_ratios=[2, 1] if show_resolution else [1],
    )

    axs = []

    ax_energy = fig.add_subplot(gs[0])
    axs.append(ax_energy)

    if show_ratio:
        ax_ratio = fig.add_subplot(gs[1 + show_resolution])
        axs.append(ax_ratio)

    if show_resolution:
        ax_resolution = fig.add_subplot(gs[:, 1])
        axs.append(ax_resolution)

    _logger.debug(
        "Created a spectrum plot layout. "
        + ("+ MC-Data ratio plot. " if show_ratio else "")
        + ("+ Plot of energy resolution." if show_resolution else "")
    )

    return fig, _ensure_axs_tuple(axs=axs)


def fd_uv_views_layout(
    **figure_kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    """\
    Create a custom layout for the far detector U-Z and V-Z plane views.

    Returns
    -------
    tuple[Figure, tuple[Axes, ...]]
        Matplotlib `Figure` object and a tuple of Matplotlib `Axes` object(s).
    """
    axs_edge_colour = plt.rcParams["axes.edgecolor"]

    fd_depths = np.asarray(
        [
            minos_numbers["FD"]["South"]["D"],
            minos_numbers["FD"]["AirGap"]["D"],
            minos_numbers["FD"]["North"]["D"],
        ]
    )

    fd_depth_ratios = fd_depths / fd_depths.sum()

    fig, axs = grid_layout(
        n_rows=2,
        n_cols=3,
        constrained_layout=False,
        width_ratios=fd_depth_ratios,
        **figure_kwargs,
    )

    fig.subplots_adjust(wspace=0)

    # Indicating the air gap...
    axs[1].add_patch(
        patches.Rectangle(
            (0, 0),
            1,
            1,
            linewidth=0.5,
            edgecolor=axs_edge_colour,
            facecolor="none",
            hatch="//",
        )
    )
    axs[4].add_patch(
        patches.Rectangle(
            (0, 0),
            1,
            1,
            linewidth=0.5,
            edgecolor=axs_edge_colour,
            facecolor="none",
            hatch="//",
        )
    )

    axs[0].text(0.07, 0.87, "U-Z Plane".upper(), transform=axs[0].transAxes)
    axs[3].text(0.07, 0.87, "V-Z Plane".upper(), transform=axs[3].transAxes)

    x_label = "Plane Number".upper()
    y_label = "Strip Number".upper()

    axs[0].tick_params(which="both", right=False)
    axs[0].set_ylabel(y_label)
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[2].tick_params(which="both", left=False, labelleft=False)
    axs[3].tick_params(which="both", right=False)
    axs[3].set_ylabel(y_label)
    axs[4].set_yticks([])
    axs[4].set_xticks([])
    axs[4].set_yticklabels([])
    axs[4].set_xticklabels([])
    axs[5].tick_params(which="both", left=False, labelleft=False)

    fig.supxlabel(
        x_label,
        fontsize=axs[0].xaxis.label.get_fontsize(),
        transform=axs[1].xaxis.label.get_transform(),
    )

    return fig, axs


# ============================== [ Modifiers  ] ============================== #


def energy_axs_scale(
    ax: Axes,
    segments: list[tuple[float, float, float]] | None = None,
    x_ticks: npt.ArrayLike | None = None,
    which_axis: Literal["x", "y"] = "x",
) -> None:
    """\
    Set the x-axis scale for an energy spectrum plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib `Axes` object.
    
    segments : list[tuple[float, float, float]]
        List of tuples, where each tuple contains the start, end, and step 
        of the segment. Set to default configuration if `None` is given.
        
    x_ticks : npt.ArrayLike
        Array of x-axis ticks. Set to default configuration if `None` is 
        given.
        
    which_axis : str
        Which axis to set the scale for. Defaults to "x".
    """
    if segments is None:
        segments = DEFAULT_X_AXIS_SEGMENTS

    if x_ticks is None:
        x_ticks = DEFAULT_X_AXIS_TICKS

    ax.set_xscale(
        scl.FuncScale(
            axis=(ax.xaxis if which_axis == "x" else ax.yaxis),
            functions=(
                lambda x: _axs_fwd_transform(segments=segments, array=x),
                lambda x: _axs_inv_transform(segments=segments, array=x),
            ),
        )
    )
    ax.set_xticks(x_ticks)

    _logger.debug("Modified the energy x-axis.")


def spec_fig_cleanup(
    fig: Figure,
    ax_energy: Axes,
    ax_ratio: Axes | None = None,
    ax_resolution: Axes | None = None,
) -> None:
    """\
    Cleans up the spectrum plot figure.

    Parameters
    ----------
    fig : Figure
        Matplotlib `Figure` object.

    ax_energy : Axes
        Matplotlib `Axes` object for the energy spectrum plot.

    ax_ratio : Axes
        Matplotlib `Axes` object for the ratio plot. Defaults to `None`.

    ax_resolution : Axes
        Matplotlib `Axes` object for the resolution plot. Defaults to `None`.
    """
    axs_edge_colour = plt.rcParams["axes.edgecolor"]

    if ax_ratio is not None:
        ax_ratio.axhline(
            0,
            color=axs_edge_colour,
            linestyle="dashed",
            linewidth=plt.rcParams["xtick.major.width"],
        )
        ax_energy.set_xticklabels([])
        ax_ratio.set_ylim(-1.0, 1.0)
        ax_ratio.set_yticks([-0.5, 0, 0.5])

    if ax_resolution is not None:
        ax_resolution.axvline(
            0,
            color=axs_edge_colour,
            linestyle="dashed",
            linewidth=plt.rcParams["xtick.major.width"],
        )
        ax_resolution.yaxis.set_label_position("right")
        ax_resolution.tick_params(axis="y", labelleft=False, labelright=True)
        ax_resolution.set_xlim(-1, 1)
        ax_resolution.set_xticks([-0.5, 0, 0.5])

    # Werid trick to make the plot fill the entire figure...

    fig.tight_layout()

    if ax_ratio is not None:
        fig.subplots_adjust(hspace=0.0)

    if ax_resolution is not None:
        fig.subplots_adjust(wspace=0.05)

    _logger.debug("Cleaned up the spectrum plot figure.")


# =============================== [ Plotting ] =============================== #


def plot_hist(
    data: npt.NDArray,
    bins: int | npt.NDArray,
    # Optional Figure & Axes
    fig: Figure | None = None,
    ax: Axes | None = None,
    **hist_kwargs,
) -> tuple[Figure, Axes, dict[str, float | npt.NDArray]]:
    if (fig is None) or (ax is None):
        fig, (ax, *_) = grid_layout()

    # (1) Plot the histogram.

    bin_heights, bin_edges, _ = ax.hist(
        data,
        bins=bins,  # pyright: ignore reportArgumentType
        **hist_kwargs,
    )

    # (2) Calculate histogram and stats info.

    info: dict[str, float | npt.NDArray] = {
        # Histogram
        "BinHeights": np.asarray(bin_heights, dtype=float),
        "BinEdges": np.asarray(bin_edges, dtype=float),
        "BinCenters": get_bin_centers(bin_edges=bin_edges),
        # Stats
        "Mean": float(np.mean(data)),
        "StD": float(np.std(data)),
        "Min": float(np.min(data)),
        "Max": float(np.max(data)),
        "Median": float(np.median(data)),
    }

    return fig, ax, info


def plot_hist_from_heights() -> ...:
    pass


# ============================== [ Templates  ] ============================== #


def plot_energy_resolution(
    reco_energy: npt.ArrayLike,
    mc_energy: npt.ArrayLike,
    algorithm_name: str = "",
    **figure_kwargs,
) -> tuple[Figure, tuple[Axes, ...], dict[str, float]]:
    """\
    Plot the resolution of an energy estimator.

    Parameters
    ----------
    reco_energy : npt.ArrayLike
        Reconstructed energy using the algorithm.
        
    mc_energy : npt.ArrayLike
        True energy.
        
    algorithm_name : str
        Optional: Name of the algorithm used to estimate the energy.

    Returns
    -------
    Figure
        Matplotlib `Figure` object.
        
    tuple[Axes, ...]
        Tuple of Matplotlib `Axes` object(s).
        
    dict[str, float]
        Dictionary containing the mean and standard deviation of the energy
        resolution distribution in the keys 'Mean' and 'Std' respectively.
    """
    reco_energy = np.asarray(reco_energy)
    mc_energy = np.asarray(mc_energy)

    fig, axs = spectrum_layout(
        show_ratio=True, show_resolution=True, **figure_kwargs
    )

    ax = axs[0]

    reco_bin_heights, _, _ = ax.hist(
        reco_energy,
        bins=MINOS_GUESSED_ENERGY_BINS,
        label="RECO.",
    )

    mc_bin_heights, bin_edges, _ = ax.hist(
        mc_energy,
        bins=MINOS_GUESSED_ENERGY_BINS,
        histtype="step",
        label="MC",
    )

    energy_axs_scale(ax)

    ax.set_title(algorithm_name.upper())
    ax.set_ylabel("Events".upper())
    ax.legend()

    ax = axs[1]

    ax.set_xlabel("Neutrino Energy, ".upper() + r"$E_\nu$ [GeV]")
    ax.set_ylabel("Ratio - 1".upper())

    mc_bin_heights = np.asarray(mc_bin_heights)
    reco_bin_heights = np.asarray(reco_bin_heights)

    ax.plot(
        get_bin_centers(bin_edges=bin_edges),
        (mc_bin_heights / reco_bin_heights) - 1,
        "o",
    )

    energy_axs_scale(ax)

    ax = axs[2]

    resolution = (mc_energy / reco_energy) - 1
    mean_resolution = float(np.mean(resolution))
    std_resolution = float(np.std(resolution))

    ax.hist(
        resolution,
        bins=np.linspace(-1, 1, 30),  # pyright: ignore reportArgumentType
    )

    ax.set_title(
        r"$\mu=$"
        + f"{mean_resolution:6.4f}, "
        + r"$\sigma=$"
        + f"{std_resolution:6.4f}"
    )
    ax.set_xlabel(r"$E_\nu$" + " Resolution".upper())
    ax.set_ylabel("Frequency".upper())

    spec_fig_cleanup(fig, *axs)

    _logger.debug("Using the 'Energy Estimator' template.")

    return fig, axs, {"Mean": mean_resolution, "StD": std_resolution}


def plot_fd_event_image(
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    fill: npt.NDArray | None = None,
    toggle_log_scale: bool = False,
    cbar_label: str = "",
    **figure_kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    """\
    Plot the pixel images of the event.

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

    fill : npt.NDArray | None
        Array to fill the image. Defaults to `None`. If `None`, the image will 
        be filled with "1"s.

    toggle_log_scale : bool
        Whether to use a log scale for the pixel images. Defaults to `False`.

    cbar_label : str
        Label for the colour bar. Defaults to "".

    Returns
    -------
    tuple[Figure, tuple[Axes, ...]]
        Matplotlib `Figure` object and a tuple of Matplotlib `Axes` object(s).
    """
    fd_s_n_planes: int = minos_numbers["FD"]["South"]["NPlanes"]
    fd_n_n_planes: int = minos_numbers["FD"]["North"]["NPlanes"]
    fd_n_strips = minos_numbers["FD"]["NStripsPerPlane"]

    # (1) Create the figure and axes.

    fig, axs = fd_uv_views_layout(**figure_kwargs)

    # (2) Getting the event images.

    u_south_image, u_north_image = create_fd_split_image(
        plane=EPlaneView.U,
        stp_planeview=stp_planeview,
        stp_strip=stp_strip,
        stp_plane=stp_plane,
        fill=[fill] if fill is not None else None,
    )

    v_south_image, v_north_image = create_fd_split_image(
        plane=EPlaneView.V,
        stp_planeview=stp_planeview,
        stp_strip=stp_strip,
        stp_plane=stp_plane,
        fill=[fill] if fill is not None else None,
    )

    if toggle_log_scale:
        u_south_image = np.log1p(u_south_image)
        u_north_image = np.log1p(u_north_image)
        v_south_image = np.log1p(v_south_image)
        v_north_image = np.log1p(v_north_image)

    # (3) Plotting the images.

    stacked_images = np.hstack(
        [u_north_image, u_south_image, v_south_image, v_north_image]
    )

    imshow_kwargs: dict[str, Any] = {
        "origin": "lower",
        "aspect": "auto",
        "vmin": (np.min(stacked_images) if fill is not None else None),
        "vmax": (np.max(stacked_images) if fill is not None else None),
    }
    west_extent: tuple[int, int, int, int] = (
        0,
        fd_s_n_planes,
        0,
        fd_n_strips,
    )
    east_extent: tuple[int, int, int, int] = (
        fd_s_n_planes,
        fd_s_n_planes + fd_n_n_planes,
        0,
        fd_n_strips,
    )

    image = axs[0].imshow(u_south_image, extent=west_extent, **imshow_kwargs)
    axs[2].imshow(u_north_image, extent=east_extent, **imshow_kwargs)

    axs[3].imshow(v_south_image, extent=west_extent, **imshow_kwargs)
    axs[5].imshow(v_north_image, extent=east_extent, **imshow_kwargs)

    # (4) Add the colourbar.

    if fill is not None:
        colour_bar = fig.colorbar(image, ax=axs, pad=0.02, aspect=30)
        colour_bar.set_label(
            "log(1 + " * toggle_log_scale
            + cbar_label.upper()
            + ")" * toggle_log_scale
        )

    return fig, axs


# =============================== [ Helpers  ] =============================== #


def get_bin_centers(bin_edges: npt.ArrayLike) -> npt.NDArray:
    """\
    Get the bin centers from the bin edges.

    Parameters
    ----------
    bin_edges : npt.ArrayLike
        The bin edges.

    Returns
    -------
    np.ndarray
        The bin centers.
    """
    bin_edges = np.asarray(bin_edges)

    return (bin_edges[:-1] + bin_edges[1:]) / 2
