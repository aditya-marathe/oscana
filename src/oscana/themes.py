"""\
oscana / themes.py
"""

from __future__ import annotations

__all__ = ["Theme", "themes"]

from typing import Any

from pathlib import Path

from importlib import resources

from dataclasses import dataclass

from matplotlib import cycler  # type: ignore -> Should be fine (hopefully).
from matplotlib import font_manager as fm


# ============================== [ Constants  ] ============================== #

# Note: A better way to do this would be using `importlib.resources.files` but
#       I am currenly using Python 3.8 which does not seem to have this feature.
#       The other option is to use `pkg_resources` but it is deprecated! So, for
#       now, I am using `importlib.resources.path` and going back two
#       directories to get the resources folder.

with resources.path("oscana", "") as _path:
    RESOURCES_PATH = Path(_path).parent.parent / "res"

DEFAULT_FIG_SIZE: tuple[float, float] = (7.5, 6.5)  # in

# ========================= [ Classes & Functions  ] ========================= #


@dataclass(frozen=True)
class Theme:
    """\
    Dataclass for plotting themes. Only for astheics!

    Attributes
    ----------
    edge_colour : str
        Colour of plot borders (edges) and ticks.
    face_colour : str
        Colour of the plot background ("face").
    text_colour : str
        Colour of the text.
    colour_cycle : list[str]
        List of colours used to create the plot colour cycle.
    title_size : int
        Size of the plot title. In pixels.
    text_size : int
        Size of the text. In pixels.
    """

    # Using __slots__ to (somewhat) reduce memory usage...
    __slots__ = [
        "edge_colour",
        "face_colour",
        "text_colour",
        "colour_cycle",
        "title_size",
        "text_font",
        "text_size",
    ]

    # Colours

    edge_colour: str
    face_colour: str
    text_colour: str
    colour_cycle: list[str]

    # Text

    title_size: int
    text_font: str
    text_size: int

    def get_cycler(self) -> cycler:
        """\
        Gets the `cycler` object which is used for plot colour cycles.

        Returns
        -------
        cycler
            The `cycler` object.
        """
        return cycler(c=self.colour_cycle)


def _load_font(font_name: str) -> str:
    """\
    [Internal] Loads the font from the given path.

    Parameters
    ----------
    font_name : str
        Path to the font file.

    Returns
    -------
    str
        Name of the font.
    """
    # Default font
    font = "sans-serif"

    # Check if the custom font exists.
    font_as_path = RESOURCES_PATH / "fonts" / font_name

    if font_as_path.exists():
        font_object = fm.FontProperties(fname=font_as_path)  # type: ignore
        font = font_object.get_name()

        fm.fontManager.addfont(font_as_path)

    return font


def _load_colour_cycle(): ...


def _load_settings(theme_name: str) -> dict[str, Any]:
    """\
    [Internal] Loads the settings for the given theme.

    Parameters
    ----------
    theme_name : str
        Name of the theme to load.

    Returns
    -------
    dict[str, Any]
        Dictionary of Matplotlib settings for the given theme.
    """
    theme: Theme = themes.get(theme_name, themes["draft"])

    return {
        # Quality
        "figure.dpi": 100,
        "text.antialiased": True,
        "lines.antialiased": True,
        "patch.antialiased": True,
        "image.interpolation": "antialiased",
        # Figure
        "figure.edgecolor": theme.edge_colour,
        "figure.facecolor": theme.face_colour,
        "figure.figsize": DEFAULT_FIG_SIZE,
        "figure.titlesize": int(theme.title_size * 1.5),
        "figure.autolayout": True,  # Not sure if this will break anything...
        # Axes
        "axes.facecolor": theme.face_colour,
        "axes.edgecolor": theme.edge_colour,
        "axes.prop_cycle": theme.get_cycler(),
        "axes.titlelocation": "left",
        "axes.titlesize": theme.title_size,
        "axes.labelsize": theme.text_size,
        "axes.labelpad": 10,
        "axes.labelcolor": theme.text_colour,
        "axes.formatter.use_mathtext": True,
        "axes.xmargin": 0.0,
        "axes.ymargin": 0.3,
        # Axis
        "xaxis.labellocation": "center",
        "yaxis.labellocation": "center",
        # Text
        "text.color": theme.text_colour,
        "font.size": theme.text_size,
        "font.family": _load_font(font_name=theme.text_font),
        "font.stretch": "semi-expanded",
        # X-Ticks
        "xtick.top": True,
        "xtick.bottom": True,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.major.size": 7,
        "xtick.minor.size": 3.5,
        "xtick.color": theme.edge_colour,
        "xtick.labelcolor": theme.edge_colour,
        # Y-Ticks
        "ytick.left": True,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.major.size": 7,
        "ytick.minor.size": 3.5,
        "ytick.color": theme.edge_colour,
        "ytick.labelcolor": theme.edge_colour,
        # Linestyle
        "lines.dashed_pattern": (7, 4),
        # Image
        "image.cmap": "viridis",
        "image.origin": "lower",
        "image.aspect": "auto",
        # Legend
        "legend.loc": "upper right",
        "legend.frameon": False,
        "legend.framealpha": 0.4,
        "legend.facecolor": theme.face_colour,
        "legend.borderpad": 0.5,
        "legend.fontsize": theme.text_size,
        "legend.labelcolor": theme.text_colour,
    }


# ================================ [ Themes ] ================================ #

# Note: `_colour_cycle` is temporary and it may be replaced with a JSON file in
#       the resources folder.

_colour_cycle = {
    "Light": [
        "#CC0019",
        "#016AC6",
        "#FF8201",
        "#016AC6",
        "#FFB600",
        "#016AC6",
        "#AED000",
        "#00939D",
        "#00892F",
        "#00939D",
    ],
    "Dark": [
        "#FF6879",
        "#F9E17D",
        "#F1A7DC",
        "#E59F6E",
        "#C7B2DD",
        "#92C2EA",
        "#CDEA80",
        "#B8DCD2",
    ],
}

themes = {
    "slate": Theme(
        edge_colour="#FFFFFF",
        face_colour="#1E1E1E",
        text_colour="#FFFFFF",
        colour_cycle=_colour_cycle["Dark"],
        title_size=11,
        text_font="cmuntx.ttf",
        text_size=13,
    ),
    "light": Theme(
        edge_colour="#000000",
        face_colour="#FFFFFF",
        text_colour="#000000",
        colour_cycle=_colour_cycle["Light"],
        title_size=11,
        text_font="cmunrm.ttf",
        text_size=13,
    ),
    "draft": Theme(
        edge_colour="#000000",
        face_colour="#C2C2C2",
        text_colour="#000000",
        colour_cycle=_colour_cycle["Light"],
        title_size=10,
        text_font="cmunorm.ttf",
        text_size=13,
    ),
}
