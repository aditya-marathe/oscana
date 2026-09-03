"""\
oscana / themes.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

This module contains all theme-related tools for the plotting module. The
`_load_font` returns a dictionary of Matplotlib settings for the given theme,
and contains some default setting for Matplotlib `Figure`'s and `Axes`'s.
"""

from __future__ import annotations

__all__ = ["Theme", "themes"]

from typing import Any, Final

import logging

from dataclasses import dataclass

from matplotlib import cycler  # pyright: ignore reportAttributeAccessIssue
from matplotlib import font_manager as fm
from .constants import RESOURCES_PATH

# =============================== [ Logging  ] =============================== #

logger = logging.getLogger("Plot")

# ============================== [ Constants  ] ============================== #

DEFAULT_FIG_SIZE: Final[tuple[float, float]] = (9.0, 6.0)  # inches

MINIMUM_TEXT_SIZE: Final[int] = 15

CMU_BRIGHT_SEMIBOLD: Final[str] = "cmunbsr.ttf"
CMU_TYPEWRITER_TEXT_REGULAR: Final[str] = "cmuntt.ttf"

HELVITICA_RM: Final[str] = "arialr.ttf"
HELVITICA_IT: Final[str] = "ariali.ttf"
HELVITICA_BF: Final[str] = "arialb.ttf"
HELVITICA_BF_IT: Final[str] = "arialbi.ttf"

# =========================== [ Theme Dataclass  ] =========================== #


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

    # Colours

    edge_colour: str
    face_colour: str
    axes_colour: str
    text_colour: str
    colour_cycle: list[str]

    # Text

    text_font: str
    text_size: int

    # Image
    
    cmap: str

    # Math Text

    math_font_rm: str = HELVITICA_RM
    math_font_it: str = HELVITICA_IT
    math_font_bf: str = HELVITICA_BF
    math_font_bf_it: str = HELVITICA_BF_IT

    def get_cycler(self) -> cycler:
        """\
        Gets the `cycler` object which is used for plot colour cycles.

        Returns
        -------
        cycler
            The `cycler` object.
        """
        return cycler(c=self.colour_cycle)


# =========================== [ Helper Functions ] =========================== #


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

    # Check avalible fonts.
    available_fonts: set = {f.name for f in fm.fontManager.ttflist}

    if font_name in available_fonts:

        return font_name

    # Check if the custom font exists.
    font_as_path = RESOURCES_PATH / "fonts" / font_name

    if font_name.endswith(".ttf") and (not font_as_path.exists()):
        logger.warning(
            f"Font '{font_name}' not found. Defaulting to '{font.capitalize()}'"
            " font."
        )

        return font

    font_object = fm.FontProperties(fname=font_as_path)  # type: ignore
    font = font_object.get_name()

    fm.fontManager.addfont(font_as_path)

    logger.debug(f"Loaded '{font_name}' font to Matplotlib.")

    return font


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
    theme = themes.get(theme_name, None)

    if theme is None:
        logger.warning(
            f"Theme '{theme_name}' not found. Defaulting to the 'Draft' theme."
        )
        theme = themes["draft"]

    logger.debug(f"Loaded '{theme_name}' theme settings.")

    default_font = _load_font(font_name=theme.text_font)

    math_font_rm = _load_font(font_name=theme.math_font_rm)
    math_font_it = _load_font(font_name=theme.math_font_it)
    math_font_bf = _load_font(font_name=theme.math_font_bf)
    math_font_bf_it = _load_font(font_name=theme.math_font_bf_it)

    return {
        # Quality
        "figure.dpi": 130,
        "text.antialiased": True,
        "lines.antialiased": True,
        "patch.antialiased": True,
        "image.interpolation": "antialiased",
        # Figure
        "figure.edgecolor": theme.edge_colour,
        "figure.facecolor": theme.face_colour,
        "figure.figsize": DEFAULT_FIG_SIZE,
        "figure.titlesize": int(theme.text_size * 1.1),
        "figure.autolayout": True,  # Not sure if this will break anything...
        # Axes
        "axes.facecolor": theme.axes_colour,
        "axes.edgecolor": theme.edge_colour,
        "axes.prop_cycle": theme.get_cycler(),
        "axes.titlelocation": "left",
        "axes.titlesize": int(theme.text_size * 0.9),
        "axes.labelsize": theme.text_size,
        "axes.labelpad": 10,
        "axes.labelcolor": theme.text_colour,
        "axes.formatter.use_mathtext": True,
        "axes.formatter.limits": (-3, 3),
        "axes.xmargin": 0.0,
        "axes.ymargin": 0.3,
        # Grid
        "grid.alpha": 0.3,
        # Axis
        "xaxis.labellocation": "center",
        "yaxis.labellocation": "center",
        # Text
        "text.color": theme.text_colour,
        "font.size": theme.text_size,
        "font.family": default_font,
        "font.stretch": "semi-expanded",
        # Math Text
        "mathtext.fontset": "custom",
        "mathtext.sf": math_font_rm,
        "mathtext.rm": math_font_rm,
        "mathtext.it": f"{math_font_it}:italic",
        "mathtext.bf": f"{math_font_bf}:bold",
        "mathtext.tt": "monospace",
        "mathtext.cal": "DejaVu Sans",
        "mathtext.bfit": f"{math_font_bf_it}:bold:italic",
        "mathtext.fallback": "cm",
        "mathtext.default": "it",
        # X-Ticks
        "xtick.top": True,
        "xtick.bottom": True,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.major.size": 7,
        "xtick.minor.size": 3.5,
        "xtick.color": theme.edge_colour,
        "xtick.labelcolor": theme.text_colour,
        # Y-Ticks
        "ytick.left": True,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.major.size": 7,
        "ytick.minor.size": 3.5,
        "ytick.color": theme.edge_colour,
        "ytick.labelcolor": theme.text_colour,
        # Linestyle
        "lines.dashed_pattern": (7, 4),
        "lines.linewidth": 1.5,
        "lines.antialiased": True,
        # Image
        "image.cmap": theme.cmap,
        "image.origin": "lower",
        "image.aspect": "auto",
        # Legend
        "legend.loc": "upper right",
        "legend.frameon": True,
        "legend.framealpha": 0.4,
        "legend.facecolor": theme.face_colour,
        "legend.edgecolor": theme.face_colour,
        "legend.borderpad": 0.5,
        "legend.fontsize": int(theme.text_size * 0.9),
        "legend.labelcolor": theme.text_colour,
    }


# ================================ [ Themes ] ================================ #

# Note: `_colour_cycle` is temporary and it may be replaced with a JSON file in
#       the resources folder.

_colour_cycles = {
    "BrickPlot": [
        # A LEGO twist on the classic Matplotlib theme.
        "#0055BF",  # Blue
        "#FE8A18",  # Orange
        "#237841",  # Green
        "#C91A09",  # Red
        "#81007B",  # Purple
        "#583927",  # Brown
        "#FC97AC",  # Pink
        "#9BA19D",  # Light Gray
        "#9B9A5A",  # Olive Green
        "#008F9B",  # Dark Turquoise
    ],
    "SandBrickPlot": [
        # A pastel LEGO twist on the classic Matplotlib theme.
        "#6074A1",  # Sand Blue
        "#FA9C1C",  # Earth Orange
        "#C7D23C",  # Medium Lime
        "#D67572",  # Sand Red
        "#845E84",  # Sand Purple
        "#B67B50",  # Fabuland Brown
        "#E4ADC8",  # Bright Pink
        "#E4CD9E",  # Tan
        "#73DCA1",  # Medium Green
        "#55A5AF",  # Light Turquoise
    ],
    "ROOT": [
        "#1845FB", # kP8Blue
        "#FF5e02", # kP8Orange
        "#C91F16", # kP8Red
        "#C849A9", # kP8Pink
        "#ADAD7D", # kP8Green
        "#86C8DD", # kP8Cyan
        "#578DFF", # kP8Azure
        "#656364", # kP8Gray
    ],
}

themes = {
    "slate": Theme(
        edge_colour="#FFFFFF",  # White
        face_colour="#1E1E1E",  # Slate
        axes_colour="#1E1E1E",  # Slate
        text_colour="#FFFFFF",  # White
        colour_cycle=_colour_cycles["BrickPlot"],
        text_font=CMU_BRIGHT_SEMIBOLD,
        text_size=MINIMUM_TEXT_SIZE,
        cmap="magma",
    ),
    "sandyslate": Theme(
        edge_colour="#FFFFFF",  # White
        face_colour="#1E1E1E",  # Slate
        axes_colour="#1E1E1E",  # Slate
        text_colour="#FFFFFF",  # White
        colour_cycle=_colour_cycles["SandBrickPlot"],
        text_font=CMU_BRIGHT_SEMIBOLD,
        text_size=MINIMUM_TEXT_SIZE,
        cmap="magma",
    ),
    "light": Theme(
        edge_colour="#000000",  # Black
        face_colour="#FFFFFF",  # White
        axes_colour="#FFFFFF",  # White
        text_colour="#000000",  # Black
        colour_cycle=_colour_cycles["BrickPlot"],
        text_font=CMU_BRIGHT_SEMIBOLD,
        text_size=MINIMUM_TEXT_SIZE,
        cmap="magma_r",
    ),
    "draft": Theme(
        edge_colour="#000000",
        face_colour="#D9D9D9",
        axes_colour="#D9D9D9",
        text_colour="#000000",
        colour_cycle=_colour_cycles["BrickPlot"],
        text_font=CMU_TYPEWRITER_TEXT_REGULAR,
        text_size=MINIMUM_TEXT_SIZE,
        cmap="magma_r",
    ),
    "nova": Theme(
        edge_colour="#000000",  # Black
        face_colour="#FFFFFF",  # White
        axes_colour="#FFFFFF",  # White
        text_colour="#000000",  # Black
        colour_cycle=_colour_cycles["ROOT"],
        text_font=HELVITICA_RM,
        text_size=20,
        cmap="cividis",
    ),
    "nova-nu26-darkblue": Theme(
        edge_colour="#000000",  # Black
        face_colour="#0D79CA",  # Dark Blue
        axes_colour="#FFFFFF",  # White
        text_colour="#FFFFFF",  # White
        colour_cycle=_colour_cycles["ROOT"],
        text_font=HELVITICA_RM,
        text_size=20,
        cmap="cividis",
    ),
    "nova-nu26-lightblue": Theme(
        edge_colour="#000000",  # Black
        face_colour="#ECF6FE",  # Light Blue
        axes_colour="#FFFFFF",  # White
        text_colour="#000000",  # Black
        colour_cycle=_colour_cycles["ROOT"],
        text_font=HELVITICA_RM,
        text_size=20,
        cmap="cividis",
    ),
}
