"""\
oscana / evd / app.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------
"""

__all__ = ["EventDisplayApp"]

import tkinter as tk

try:
    from ..themes import themes
except ImportError:
    from oscana.themes import themes


class EventDisplayApp(tk.Tk):
    def __init__(self, theme_name: str = "slate") -> None:
        super().__init__()

        self.theme = themes.get(theme_name, themes["draft"])

        self.title("Event Display")
        self.WIDTH, self.HEIGHT = 800, 500

        # screen_width = self.winfo_screenwidth()
        # screen_height = self.winfo_screenheight()
        # y_pos = int(0.5 * (screen_height - self.HEIGHT))
        # x_pos = int(0.5 * (screen_width - self.WIDTH))

        y_pos, x_pos = 0, 0

        self.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x_pos}+{y_pos}")

        # Tkinter Variables / Config

        self.config(bg=self.theme.face_colour)

        self.header_label = tk.StringVar(self)
        self.time_label = tk.StringVar(self)
        self.dropdown = tk.StringVar(self)

        _text_config = {
            "bg": self.theme.face_colour,
            "fg": self.theme.text_colour,
            "anchor": tk.W,
            "justify": tk.LEFT,
        }

        _button_config = {
            "bg": self.theme.face_colour,
            "fg": self.theme.text_colour,
        }

        _pack_config = {
            "anchor": tk.W,
            "side": tk.TOP,
            "fill": tk.X,
            "expand": False,
            "padx": 5,
            "pady": 5,
        }

        # Widgets

        tk.Label(self, textvariable=self.header_label, cnf=_text_config).pack(
            cnf=_pack_config
        )
        tk.Label(self, textvariable=self.time_label, cnf=_text_config).pack(
            cnf=_pack_config
        )

        # Plot Container

        plot_container = tk.Frame(self, bg=self.theme.face_colour)
        plot_container.pack(cnf=_pack_config)

        # Bottom Container

        bottom_container = tk.Frame(self, bg=self.theme.face_colour)
        bottom_container.pack(side=tk.BOTTOM, padx=0, pady=0, cnf=_pack_config)

        tk.Button(bottom_container, text="Prev", cnf=_button_config).pack(
            side=tk.LEFT
        )
        tk.Button(bottom_container, text="Next", cnf=_button_config).pack(
            side=tk.RIGHT
        )

        self.update_display()

    def update_display(self) -> None:
        self.header_label.set("Hello World")
        self.time_label.set("Hello World")


def get_event_images() -> ...: ...


if __name__ == "__main__":
    EventDisplayApp().mainloop()
