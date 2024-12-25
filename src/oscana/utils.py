"""\
oscana / utils.py
"""

__all__ = [
    "OscanaError",
    "load_env_file",
]

import dotenv

# ============================== [ Exceptions ] ============================== #


class OscanaError(Exception):
    pass


# ============================== [ Functions  ] ============================== #


def load_env_file() -> None:
    """\
    Load the .env file in the root directory of the project.
    """
    if not dotenv.load_dotenv():
        raise OscanaError("Unsuccessful in loading .env file!")
