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
    "marginal_hist_layout",
    # Modifiers
    "energy_axs_scale",
    "spec_fig_cleanup",
    "add_experiment_tag",
    # Plotting
    "plot_hist",
    "plot_hist_from_heights",
    # Templates
    "plot_energy_resolution",
    "plot_fd_event_image",
    "plot_2d_hist",
]

from typing import Generator, Literal, Any, TYPE_CHECKING, TypeAlias

from contextlib import contextmanager

import logging, warnings

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.scale as scl
from matplotlib import patches
from matplotlib import offsetbox

from .themes import _load_settings
from .utils import minos_numbers, get_bin_centers
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

ExperimentType: TypeAlias = Literal["MINOS", "MINOS+", "NOvA"]
PlotAccessType: TypeAlias = Literal[
    "Internal",  # Private
    "Preliminary",  # Public Preliminary
    "Simulation",  # Public Preliminary
    "Data",  # Public Preliminary
    "Final",  # Public Final
    "Unknown",  # ???
]

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
    original_axis = np.zeros_like(array, dtype=float)

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
    original_params = {k: mpl.rcParams[k] for k in settings}

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
    title: str = "Far Detector",
    use_real_distances: bool = False,
    **figure_kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    """\
    Create a custom layout for the far detector U-Z and V-Z plane views.

    Parameters
    ----------
    title : str
        Title of the plot. Defaults to "Far Detector".

    use_real_distances : bool
        Whether to use real distances for the axes labels. Defaults to `False`.

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

    plane_label_x_pos = 0.07
    plane_label_y_pos = 0.87

    axs[0].text(
        plane_label_x_pos,
        plane_label_y_pos,
        "UZ Plane",
        fontdict=dict(fontsize=mpl.rcParams["axes.titlesize"]),
        transform=axs[0].transAxes,
    )
    axs[3].text(
        plane_label_x_pos,
        plane_label_y_pos,
        "VZ Plane",
        fontdict=dict(fontsize=mpl.rcParams["axes.titlesize"]),
        transform=axs[3].transAxes,
    )

    x_label = "Z (m)" if use_real_distances else "Plane Number"
    y_label = "T (m)" if use_real_distances else "Strip Number"

    axs[0].tick_params(which="both", right=False)
    axs[0].set_ylabel(y_label)
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[2].tick_params(which="both", left=False, labelleft=False)
    axs[2].set_title(title, loc="right", wrap=True)
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

    fig.subplots_adjust(bottom=0.08)

    return fig, axs


def marginal_hist_layout(
    gridspec_kwargs: dict[str, Any] | None = None, **figure_kwargs: Any
) -> tuple[Figure, tuple[Axes, ...]]:
    """\
    Create a custom layout for a 2D histogram with marginal histograms.

    Parameters
    ----------
    gridspec_kwargs : dict[str, Any] | None
        Additional keyword arguments to pass to the `GridSpec` constructor.
    
    **figure_kwargs : Any
        Additional keyword arguments to pass to the `plt.figure` function.

    Returns
    -------
    tuple[Figure, tuple[Axes, ...]]
        Matplotlib `Figure` object and a tuple of Matplotlib `Axes` object(s).
    """
    if gridspec_kwargs is None:
        gridspec_kwargs = {}

    if "figsize" in figure_kwargs:
        width = max(figure_kwargs.pop("figsize"))
    else:
        width = max(plt.rcParams["figure.figsize"])

    fig = plt.figure(figsize=(width, width), **figure_kwargs)

    gs = gridspec.GridSpec(
        width_ratios=[3, 1],
        height_ratios=[1, 3],
        hspace=0.05,
        wspace=0.05,
        **gridspec_kwargs,
        nrows=2,
        ncols=2,
    )

    ax = fig.add_subplot(gs[1, 0])

    ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_x.tick_params(axis="x", labelbottom=False)

    ax_y = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_y.margins(x=plt.rcParams["axes.ymargin"], y=plt.rcParams["axes.xmargin"])
    ax_y.xaxis.tick_top()
    ax_y.tick_params(axis="y", labelleft=False)

    return fig, (ax, ax_x, ax_y)


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


def add_experiment_tag(
    ax: Axes,
    rel_x_pos: float = 0.0,
    rel_y_pos: float = 1.21,
    experiment: ExperimentType = "MINOS",
    access: PlotAccessType = "Unknown",
    extra_tags: list[str] | None = None,
    flatten_tag: bool = False,
    **_options: Any,
) -> None:
    """\
    Add an experiment tag to the plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib `Axes` object to add the tag to.

    rel_x_pos : float
        Relative x position of the tag in the axes coordinates. Defaults to 0.0.
    
    rel_y_pos : float
        Relative y position of the tag in the axes coordinates. Defaults to 
        1.21.

    experiment : ExperimentType
        Name of the experiment. Defaults to "MINOS".
    
    access : PlotAccessType
        Access type of the plot. Defaults to "Unknown".

    extra_tags : list[str] | None
        Extra tags to add to the plot. Defaults to `None`.

    flatten_tag : bool
        Whether to flatten the extra tags into a single line. Defaults to 
        `False`.

    **_options : Any
        Optional keyword arguments to customize the tag appearance.

    Notes
    -----
    If `fig` is specified, the tag will be added to the figure instead of the
    axes.

    Optional keyword arguments include:
        - `loc`: Location of the tag in the axes coordinates.
        - `h_padding`: Horizontal padding between the components.
        - `h_align`: Horizontal alignment of the components.
        - `h_separation`: Horizontal separation between the components.
        - `v_padding`: Vertical padding between the components.
        - `v_align`: Vertical alignment of the components.
        - `v_separation`: Vertical separation between the components.
        - `padding`: Padding around the tag.
        - `in_padding`: Padding inside the tag.
        - `enable_frame`: Whether to enable the frame around the tag.
    """
    title_props = {"fontsize": mpl.rcParams["font.size"]}
    extra_props = {"fontsize": mpl.rcParams["legend.fontsize"]}

    experiment_str = r"$\mathbf{" + experiment + "}$"

    if str(experiment).lower() == "nova":
        experiment_str = r"$\mathbf{NO \nu A}$"

    components = [
        [
            offsetbox.TextArea(experiment_str, textprops=title_props),
            offsetbox.TextArea(
                (
                    ""
                    if (access == "Unknown") or (access == "Final")
                    else r"$\mathit{" + access + "}$"
                ),
                textprops=title_props,
            ),
        ]
    ]

    if extra_tags:
        if flatten_tag:
            components[0].extend(
                [
                    offsetbox.TextArea(" | ", textprops=title_props),
                    offsetbox.TextArea(
                        " \u2022 ".join(extra_tags), textprops=extra_props
                    ),
                ]
            )
        else:
            components.append(
                [
                    offsetbox.TextArea(
                        "\n".join(extra_tags), textprops=extra_props
                    ),
                ]
            )

    h_components: list[offsetbox.HPacker] = []

    for j in range(len(components)):
        if components[j]:
            h_components.append(
                offsetbox.HPacker(
                    children=(
                        components[j]
                    ),  # pyright: ignore[reportArgumentType]
                    pad=_options.get("h_padding", 0.0),
                    align=_options.get("h_align", "left"),
                    sep=_options.get("h_separation", 2.0),
                )
            )

    container = offsetbox.AnchoredOffsetbox(
        loc=_options.get("loc", "upper left"),
        child=offsetbox.VPacker(
            children=h_components,  # pyright: ignore[reportArgumentType]
            pad=_options.get("v_padding", 0.0),
            align=_options.get("v_align", "left"),
            sep=_options.get("v_separation", 2.0),
        ),
        pad=_options.get("padding", 0.0),
        borderpad=_options.get("in_padding", 0.0),
        frameon=_options.get("enable_frame", False),
        bbox_to_anchor=(rel_x_pos, rel_y_pos),
        bbox_transform=ax.transAxes,
    )

    ax.add_artist(container)


# =============================== [ Plotting ] =============================== #


def plot_hist(
    data: npt.NDArray,
    bins: int | npt.NDArray | list[float],
    # Optional Figure & Axes
    fig: Figure | None = None,
    ax: Axes | None = None,
    **hist_kwargs,
) -> tuple[Figure, Axes, dict[str, float | npt.NDArray]]:
    """\
    Plot a histogram of the data.

    Parameters
    ----------
    data : npt.NDArray
        The data.

    bins : int | npt.NDArray | list[float]
        The number of bins or the bin edges.

    fig : Figure | None
        Optional: Matplotlib `Figure` object. If `None`, a new figure will be
        created.
    
    ax : Axes | None
        Optional: Matplotlib `Axes` object. If `None`, a new axes will be
        created.
    
    hist_kwargs : dict[str, Any]
        Optional: Additional keyword arguments to pass to the `hist` function.

    Returns
    -------
    tuple[Figure, Axes, dict[str, float | npt.NDArray]]
        Matplotlib `Figure` object, Matplotlib `Axes` object, and a dictionary
        containing the histogram and statistics information.
    """
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


def plot_hist_from_heights(
    bin_heights: npt.NDArray,
    bins: npt.NDArray | list[float],
    # Optional Figure & Axes
    fig: Figure | None = None,
    ax: Axes | None = None,
    **hist_kwargs: Any,
) -> tuple[Figure, Axes, dict[str, float | npt.NDArray]]:
    """\
    Plot a histogram from the bin heights and edges.

    Parameters
    ----------
    bin_heights : npt.NDArray
        The heights of the bins.

    bins : npt.NDArray | list[float]
        The bin edges.

    fig : Figure | None
        Optional: Matplotlib `Figure` object. If `None`, a new figure will be
        created.

    ax : Axes | None
        Optional: Matplotlib `Axes` object. If `None`, a new axes will be
        created.

    **hist_kwargs : Any
        Optional: Additional keyword arguments to pass to the `hist` function.

    Returns
    -------
    tuple[Figure, Axes, dict[str, float | npt.NDArray]]
        Matplotlib `Figure` object, Matplotlib `Axes` object, and a dictionary
        containing the histogram and statistics information.

    Notes
    -----
    The info dictionary do not contain the statistics of the data, as it is not
    provided. It only contains the histogram information.
    """

    # bin_centers = get_bin_centers(bin_edges=bins)

    if (fig is None) or (ax is None):
        fig, (ax, *_) = grid_layout()

    # (1) Plot the histogram.

    bin_heights, bin_edges, _ = ax.hist(
        bins[:-1],
        bins=bins,  # pyright: ignore reportArgumentType
        weights=bin_heights,
        **hist_kwargs,
    )

    # (2) Calculate histogram and stats info.

    centres = get_bin_centers(bin_edges=bin_edges)
    sum_bin_heights = np.sum(bin_heights)
    mean = np.divide(
        np.sum(centres * bin_heights),
        sum_bin_heights,
        out=np.zeros_like(sum_bin_heights, dtype=float),
        where=sum_bin_heights != 0,
    )

    info: dict[str, float | npt.NDArray] = {
        # Histogram
        "BinHeights": np.asarray(bin_heights, dtype=float),
        "BinEdges": np.asarray(bin_edges, dtype=float),
        "BinCenters": centres,
        # Stats (Estimated)
        "Mean": float(mean),
        "StD": float(
            np.sqrt(
                np.divide(
                    np.sum(bin_heights * (centres - mean) ** 2),
                    sum_bin_heights,
                    out=np.zeros_like(sum_bin_heights, dtype=float),
                    where=sum_bin_heights != 0,
                )
            )
        ),
    }

    return fig, ax, info


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

    ax.set_title(algorithm_name)
    ax.set_ylabel("Events")
    ax.legend()

    ax = axs[1]

    ax.set_xlabel("Neutrino Energy, " + r"$E_\nu$ [GeV]")
    ax.set_ylabel("Ratio - 1")

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
    ax.set_xlabel(r"$E_\nu$" + " Resolution")
    ax.set_ylabel("Frequency")

    spec_fig_cleanup(fig, *axs)

    _logger.debug("Using the 'Energy Estimator' template.")

    return fig, axs, {"Mean": mean_resolution, "StD": std_resolution}


def plot_fd_event_image(
    stp_planeview: npt.NDArray,
    stp_strip: npt.NDArray,
    stp_plane: npt.NDArray,
    fill: npt.NDArray | None = None,
    *,
    use_log_scale: bool = False,
    cbar_label: str = "???",
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

    use_log_scale : bool
        Whether to use a log scale for the pixel images. Defaults to `False`.

    cbar_label : str
        Label for the colour bar. Defaults to "???".

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

    if use_log_scale:
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
    axs[1].set_facecolor(plt.get_cmap(mpl.rcParams["image.cmap"])(0.0))
    axs[2].imshow(u_north_image, extent=east_extent, **imshow_kwargs)

    axs[3].imshow(v_south_image, extent=west_extent, **imshow_kwargs)
    axs[4].set_facecolor(plt.get_cmap(mpl.rcParams["image.cmap"])(0.0))
    axs[5].imshow(v_north_image, extent=east_extent, **imshow_kwargs)

    # (4) Add the colourbar.

    if fill is not None:
        colour_bar = fig.colorbar(image, ax=axs, pad=0.02, aspect=30)
        colour_bar.set_label(
            "log(1 + " * use_log_scale + cbar_label + ")" * use_log_scale
        )

    return fig, axs


def plot_2d_hist(
    x: npt.NDArray,
    y: npt.NDArray,
    x_bins: int | npt.NDArray | list[float] = 10,
    y_bins: int | npt.NDArray | list[float] = 10,
    z_label: str | None = None,
    show_counts: bool = False,
    add_colour_bar: bool = False,
    figure_kwargs: dict[str, Any] | None = None,
    hist_2d_kwargs: dict[str, Any] | None = None,
    hist_kwargs: dict[str, Any] | None = None,
):
    """\
    Plot a 2D histogram with marginal histograms.
    
    Parameters
    ----------
    x : npt.NDArray
        The x data.
        
    y : npt.NDArray
        The y data.
        
    x_bins : int | npt.NDArray | list[float]
        The number of bins or the bin edges for the x data. Defaults to 10.
        
    y_bins : int | npt.NDArray | list[float]
        The number of bins or the bin edges for the y data. Defaults to 10.
        
    z_label : str | None
        Label for the z-axis (colour bar). Defaults to `None`.
        
    show_counts : bool
        Whether to show the counts on the marginal histograms. Defaults to
        `True`.
        
    add_colour_bar : bool
        Whether to add a colour bar to the plot. Defaults to `False`.
        
    figure_kwargs : dict[str, Any] | None
        Additional keyword arguments to pass to the `figure` function. Defaults
        to `None`.

    hist_2d_kwargs : dict[str, Any] | None
        Additional keyword arguments to pass to the `hist2d` function. Defaults
        to `None`.

    hist_kwargs : dict[str, Any] | None
        Additional keyword arguments to pass to the `hist` function for the
        marginal histograms. Defaults to `None`.
        """
    if figure_kwargs is None:
        figure_kwargs = {}

    if hist_2d_kwargs is None:
        hist_2d_kwargs = {}

    if hist_kwargs is None:
        hist_kwargs = {}

    fig, axs = marginal_hist_layout(**figure_kwargs)

    ax, ax_x, ax_y = axs

    _, _, _, mesh = ax.hist2d(x=x, y=y, bins=(x_bins, y_bins), **hist_2d_kwargs)

    ax_x.tick_params("y", labelleft=show_counts)
    _, _, stats_x = plot_hist(
        data=x, bins=x_bins, fig=fig, ax=ax_x, **hist_kwargs
    )

    ax_y.tick_params(
        axis="x", which="both", labeltop=show_counts, bottom=True, rotation=-90
    )
    _, _, stats_y = plot_hist(
        data=y,
        bins=y_bins,
        fig=fig,
        ax=ax_y,
        **hist_kwargs,
        orientation="horizontal",
    )

    if z_label:
        ax_x.set_ylabel(z_label)
        ax_y.set_xlabel(z_label)

    if add_colour_bar:
        colour_bar = fig.colorbar(mesh, ax=ax_y, orientation="vertical")

        if z_label:
            colour_bar.set_label(z_label)

    fig.tight_layout()

    return (fig, axs, stats_x, stats_y)
