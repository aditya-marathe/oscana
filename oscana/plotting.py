"""
oscana / plot.py
"""

from __future__ import annotations

__all__ = [
    "MINOS_GUESSED_ENERGY_BINS",
    "context",
    "layout",
    "modifiers",
    "plot",
    "template",
    "helpers",
]

# Note: `layout`, `modifiers`, and `plot` are all (data) classes to organise
#       functions. In every senario using a dictionary is faster, but we
#       sacrifice the code readability. Since we are only plotting, it is ok to
#       sacrifice speed for readability.
#
#       Usually, I think, it is not a good idea to hide this kind of complexity
#       and it is better to use dictionaries.

from typing import Generator, Literal, TYPE_CHECKING

from contextlib import contextmanager

import warnings

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.scale as scl

from oscana.themes import _load_settings

if TYPE_CHECKING:
    import numpy.typing as npt

    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

# ============================== [ Constants  ] ============================== #

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

# =============================== [ Context  ] =============================== #


@contextmanager
def context(theme_name: str = "slate") -> Generator[None, None, None]:
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

    try:
        yield  # Here, we are inside the context...
    finally:
        # Overwrite the rcParameters to their original values
        mpl.rcParams.update(original_params)

        # Unfilter warnings
        warnings.simplefilter("default", UserWarning)


# =============================== [ Layouts  ] =============================== #


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
    if isinstance(axs, np.ndarray):
        axs_return = tuple(axs.flatten().tolist())
    elif isinstance(axs, list):
        axs_return = tuple(axs)
    else:
        # Note: I am going to assume that `axs` will always be one of the three
        # expected types.
        axs_return = (axs,)

    return axs_return


class _Layout:
    __slots__ = []

    @staticmethod
    def grid(
        n_rows: int = 1,
        n_cols: int = 1,
        share_x: bool = False,
        share_y: bool = False,
        **figure_kwargs,
    ) -> tuple[Figure, tuple[Axes, ...]]:
        """\
        Create a grid layout for the plot.

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
        Figure
            Matplotlib `Figure` object.
        tuple[Axes, ...]
            Tuple of Matplotlib `Axes` object(s).


        Notes
        -----
        A very lazy wrapper around `plt.subplots`.
        """
        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            sharex=share_x,
            sharey=share_y,
            **figure_kwargs,
        )

        return fig, _ensure_axs_tuple(axs=axs)

    @staticmethod
    def spec(
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
        Figure
            Matplotlib `Figure` object.
        tuple[Axes, ...]
            Tuple of Matplotlib `Axes` object(s).
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

        return fig, _ensure_axs_tuple(axs=axs)


layout = _Layout()

# ============================== [ Modifiers  ] ============================== #


def _fwd_transform(
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

    for segment in segments:
        idx = (array > segment[0]) & (array <= segment[1])
        transformed_axis[idx] = offset + (array[idx] - segment[0]) * (
            segment[2] / (segment[1] - segment[0])
        )

        offset += segment[2]

    return transformed_axis


def _inv_transform(
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

    return original_axis


class _Modifiers:
    @staticmethod
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
            List of tuples, where each tuple contains the start, end, and step of
            the segment. Set to default configuration if `None` is given.
        x_ticks : npt.ArrayLike
            Array of x-axis ticks. Set to default configuration if `None` is given.
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
                    lambda x: _fwd_transform(segments=segments, array=x),
                    lambda x: _inv_transform(segments=segments, array=x),
                ),
            )
        )
        ax.set_xticks(x_ticks)

    @staticmethod
    def spec_fig_cleanup(
        fig: Figure,
        ax_energy: Axes,
        ax_ratio: Axes | None = None,
        ax_resolution: Axes | None = None,
    ) -> None:

        if ax_ratio is not None:
            ax_ratio.axhline(
                0,
                color=plt.rcParams["axes.edgecolor"],
                linestyle="dashed",
                linewidth=plt.rcParams["xtick.major.width"],
            )
            ax_energy.set_xticklabels([])
            ax_ratio.set_ylim(-1.0, 1.0)
            ax_ratio.set_yticks([-0.5, 0, 0.5])

        if ax_resolution is not None:
            ax_resolution.axvline(
                0,
                color=plt.rcParams["axes.edgecolor"],
                linestyle="dashed",
                linewidth=plt.rcParams["xtick.major.width"],
            )
            ax_resolution.yaxis.set_label_position("right")
            ax_resolution.tick_params(
                axis="y", labelleft=False, labelright=True
            )
            ax_resolution.set_xlim(-1, 1)
            ax_resolution.set_xticks([-0.5, 0, 0.5])

        # Werid trick to make the plot fill the entire figure...

        fig.tight_layout()

        if ax_ratio is not None:
            fig.subplots_adjust(hspace=0.0)

        if ax_resolution is not None:
            fig.subplots_adjust(wspace=0.05)


modifiers = _Modifiers()

# =============================== [ Plotting ] =============================== #


class _Plot:
    @staticmethod
    def hist() -> ...: ...


plot = _Plot()

# ============================== [ Templates  ] ============================== #


class _Template:
    @staticmethod
    def energy_estimator_resolution(
        reco_energy: npt.ArrayLike,
        mc_energy: npt.ArrayLike,
        algorithm_name: str = "",
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
            resolution distribution.
        """
        reco_energy = np.asarray(reco_energy)
        mc_energy = np.asarray(mc_energy)

        fig, axs = layout.spec(show_ratio=True, show_resolution=True)

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

        modifiers.energy_axs_scale(ax)

        ax.set_title(algorithm_name.upper())
        ax.set_ylabel("Events".upper())
        ax.legend()

        ax = axs[1]

        ax.set_xlabel("Neutrino Energy, ".upper() + r"$E_\nu$ [GeV]")
        ax.set_ylabel("Ratio - 1".upper())

        mc_bin_heights = np.asarray(mc_bin_heights)
        reco_bin_heights = np.asarray(reco_bin_heights)

        ax.plot(
            helpers.get_bin_centers(bin_edges=bin_edges),
            (mc_bin_heights / reco_bin_heights) - 1,
            "o",
        )

        modifiers.energy_axs_scale(ax)

        ax = axs[2]

        resolution = (mc_energy / reco_energy) - 1
        mean_resolution = float(np.mean(resolution))
        std_resolution = float(np.std(resolution))

        ax.hist(resolution, bins=np.linspace(-1, 1, 30))

        ax.set_title(
            r"$\mu=$"
            + f"{mean_resolution:6.4f}, "
            + r"$\sigma=$"
            + f"{std_resolution:6.4f}"
        )
        ax.set_xlabel(r"$E_\nu$" + " Resolution".upper())
        ax.set_ylabel("Frequency".upper())

        modifiers.spec_fig_cleanup(fig, *axs)

        return fig, axs, {"Mean": mean_resolution, "StD": std_resolution}


template = _Template()

# =============================== [ Helpers  ] =============================== #


class _Helpers:
    @staticmethod
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


helpers = _Helpers()
