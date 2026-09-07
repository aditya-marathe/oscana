"""\
scripts / build_h5.py

--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

Builds an HDF5 file from a JSON configuration file.
"""

from __future__ import annotations

from typing import Any, Dict, Final, List, Literal
from typing import Union, TypeAlias, get_args

__all__ = []

import sys
import json
import argparse
import importlib.util
from types import ModuleType
from pathlib import Path

import oscana
from oscana.errors import OscanaError
from oscana.constants import VariableCollection
from oscana.data import DataHandler, TransformBase

# ============================== [ Constants  ] ============================== #

_FileFormat: TypeAlias = Literal["sntp", "hdf5"]

_LOADER_METHODS: Final[Dict[_FileFormat, str]] = {
    "sntp": "from_sntp",
    "hdf5": "from_hdf5",
}

_KNOWN_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "variables",
        "files",
        "file_format",
        "make_cut_bool_table",
        "transforms_module",
        "transforms",
        "compression",
        "compression_level",
    }
)

# Default arguments...

# The lookups that a transforms module is expected to define at its bottom,
# and the prefix that tells the two of them apart.

_CUT_LOOKUP_NAME: Final[str] = "cut_lookup"
_TRANSFORM_LOOKUP_NAME: Final[str] = "transform_lookup"
_CUT_PREFIX: Final[str] = "cut_"

_DEFAULT_FILE_FORMAT: Final[_FileFormat] = "sntp"
_DEFAULT_LOGS_DIR: Final[str] = "./"
_DEFAULT_VERBOSITY: Final[str] = "WARNING"
_DEFAULT_COMPRESSION_LEVEL: Final[int] = 6

_VERBOSITY_CHOICES: Final[List[str]] = [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

_CONFIG_HELP: Final[str] = """\
Example configuration file
--------------------------

{
    "file_format": "sntp",

    "variables": [
        "IMAGE_ALL_VARIABLES",
        "NtpSt/fSomeOtherVariable"
    ],

    "files": [
        "SNTP_F21042000_0024_D5_D07_R7",
        "./some/other/file.sntp.root"
    ],

    "make_cut_bool_table": true,

    "transforms_module": "./my_transforms.py",
    "transforms": [
        {"name": "reco_energy_cut", "kwargs": {"max_energy": 3.0}},
        {"name": "cut_20260325_something_else"}
    ],

    "compression": "gzip",
    "compression_level": 6
}
"""


# =============================== [ Helpers  ] =============================== #


def _read_config(config_file: Path) -> Dict[str, Any]:
    """\
    [ Internal ] Read the JSON configuration file.
    """
    config = json.loads(config_file.read_text())

    if not isinstance(config, dict):
        raise OscanaError("The configuration file must contain an object.")

    unknown_keys = sorted(set(config) - _KNOWN_CONFIG_KEYS)

    if unknown_keys:
        raise OscanaError(
            f"Unknown key(s) {unknown_keys} in the configuration file. The "
            f"known keys are {sorted(_KNOWN_CONFIG_KEYS)}."
        )

    for key in ("variables", "files"):
        if not config.get(key):
            raise OscanaError(
                f"The configuration file must contain a non-empty '{key}'."
            )

    return config


def _get_file_format(config: Dict[str, Any]) -> _FileFormat:
    """\
    [ Internal ] Get the input file format from the configuration.
    """
    file_format = config.get("file_format", _DEFAULT_FILE_FORMAT)

    if file_format not in get_args(_FileFormat):
        raise OscanaError(
            f"Unknown 'file_format' {file_format!r}. Use one of "
            f"{sorted(_LOADER_METHODS)}."
        )

    return file_format


def _resolve_variables(entries: List[str]) -> List[str]:
    """\
    [ Internal ] Expand the "variables" entries into full variable names.
    """
    variables: List[str] = []

    for entry in entries:
        collection = getattr(oscana, entry, None)

        if isinstance(collection, VariableCollection):
            variables.extend(collection.uproot)
            continue

        if "/" not in entry:
            raise OscanaError(
                f"Variable {entry!r} has no ROOT branch attached, and is not "
                "the name of a `VariableCollection` exported by `oscana`. "
                'Write it as "<branch>/<variable>".'
            )

        variables.append(entry)

    # Note: The `DataHandler` removes duplicates itself, but warns when it
    #       does. Expanding overlapping collections is a reasonable thing to
    #       do here, so we deduplicate quietly instead.
    return list(dict.fromkeys(variables))


def _import_transforms_module(module: str) -> ModuleType:
    """\
    [ Internal ] Import the module holding the user's transform classes.
    """
    module_path = Path(module).expanduser()

    if module_path.suffix != ".py":
        return importlib.import_module(module)

    module_path = module_path.resolve()

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)

    if (spec is None) or (spec.loader is None):
        raise OscanaError(
            f"Could not import '{module_path!s}' as a Python module."
        )

    imported = importlib.util.module_from_spec(spec)

    # Note: Registering the module before executing it lets the module use
    #       things which look themselves up in `sys.modules` while the module
    #       body is still running.
    sys.modules[spec.name] = imported
    spec.loader.exec_module(imported)

    return imported


def _find_transform_class(module: ModuleType, name: str) -> type[TransformBase]:
    """\
    [ Internal ] Find a transform class in the user's module by name.
    """
    missing = [
        lookup_name
        for lookup_name in (_CUT_LOOKUP_NAME, _TRANSFORM_LOOKUP_NAME)
        if not callable(getattr(module, lookup_name, None))
    ]

    if missing:
        raise OscanaError(
            f"The transforms module '{module.__name__}' does not define "
            f"{missing}. Add them at the bottom of the file with "
            "`get_func_lookup(globals_=globals(), prefix=...)`."
        )

    # Note: A lookup covers one prefix and logs an error when it misses, so we
    #       check which prefix the name lives under rather than trying one and
    #       falling back to the other.

    is_cut = any(
        attr_name.startswith(_CUT_PREFIX) and attr_name.endswith(name)
        for attr_name in vars(module)
    )

    lookup = getattr(
        module, _CUT_LOOKUP_NAME if is_cut else _TRANSFORM_LOOKUP_NAME
    )

    return lookup(name)


def _build_transforms(config: Dict[str, Any]) -> List[TransformBase]:
    """\
    [ Internal ] Build the transforms described by the configuration.
    """
    entries = config.get("transforms", [])

    if not entries:
        return []

    module_name = config.get("transforms_module")

    if not module_name:
        raise OscanaError(
            "The configuration file has 'transforms' but no "
            "'transforms_module' to import them from."
        )

    module = _import_transforms_module(module=module_name)

    transforms: List[TransformBase] = []

    for entry in entries:
        # ~ a bare string is just a transform with no keyword arguments
        settings: Dict[str, Any] = (
            {"name": entry} if isinstance(entry, str) else entry
        )

        transform_class = _find_transform_class(
            module=module, name=settings["name"]
        )

        transforms.append(transform_class(**settings.get("kwargs", {})))

    return transforms


def _check_output_file(output_file: Union[str, Path], overwrite: bool) -> Path:
    """\
    [ Internal ] Check the output file before any of the expensive work.
    """
    resolved = Path(output_file).expanduser().resolve()

    if resolved.suffix != ".h5":
        raise OscanaError(
            f"The output file '{resolved!s}' must have a '.h5' suffix."
        )

    if resolved.is_file() and (not overwrite):
        raise OscanaError(
            f"The output file '{resolved!s}' already exists. Pass "
            "'--overwrite' to replace it."
        )

    return resolved


def _init_argsparse() -> argparse.ArgumentParser:
    """\
    [ Internal ] Build the command line argument parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_CONFIG_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_file", help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "output_file", help="Path of the HDF5 file to write (must end '.h5')."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--logs-dir",
        default=_DEFAULT_LOGS_DIR,
        help=(
            "Directory to write the Oscana logs to. Defaults to "
            f"'{_DEFAULT_LOGS_DIR}'."
        ),
    )
    parser.add_argument(
        "--verbosity",
        default=_DEFAULT_VERBOSITY,
        choices=_VERBOSITY_CHOICES,
        help=(
            "Verbosity of the console output. Defaults to "
            f"'{_DEFAULT_VERBOSITY}'."
        ),
    )
    parser.add_argument(
        "--dotenv-dir",
        default=None,
        help="Directory holding the '.env' file. Defaults to the CWD.",
    )

    return parser


# ================================= [ Main ] ================================= #


def main() -> int:
    """\
    Build an HDF5 file from a JSON configuration file.
    """
    args = _init_argsparse().parse_args()

    # (1) Check the output file first, so a mistake there does not cost the
    #     user a full load.

    output_file = _check_output_file(
        output_file=args.output_file, overwrite=args.overwrite
    )

    # (2) Read the configuration file.

    config = _read_config(
        config_file=Path(args.config_file).expanduser().resolve()
    )

    # (3) Initialise Oscana. This has to happen before the transforms module
    #     is imported, since that module will almost certainly use Oscana.

    oscana.init(
        logs_dir=args.logs_dir,
        verbosity=args.verbosity,
        dotenv_dir=args.dotenv_dir,
    )

    # (4) Work out what to load, and what to do with it.

    variables = _resolve_variables(entries=config["variables"])
    transforms = _build_transforms(config=config)

    files = config["files"]
    file_format = _get_file_format(config=config)

    print(
        f"Loading {len(variables)} variable(s) from {len(files)} "
        f"{file_format.upper()} file(s)..."
    )

    # (5) Load the files into a `DataHandler`.

    dh: DataHandler = DataHandler(
        variables=variables,
        make_cut_bool_table=bool(config.get("make_cut_bool_table", False)),
    )

    getattr(dh.io, _LOADER_METHODS[file_format])(files)

    n_rows_loaded = dh.io.get_n_rows_data_table()

    if not n_rows_loaded:
        raise OscanaError(
            "No data was loaded from the given files, so there is nothing to "
            "write. (See the warnings above.)"
        )

    print(f"Loaded {n_rows_loaded:,} rows.")

    # (6) Apply the transforms.

    if transforms:
        print(f"Applying {len(transforms)} transform(s)...")

        dh.apply_transforms(transforms=transforms)

        print(
            f"{n_rows_loaded:,} rows -> "
            f"{dh.io.get_n_rows_data_table():,} rows after the transforms."
        )

    # (7) Write the output file.

    dh.io.to_hdf5(
        out_file=output_file,
        compression=config.get("compression"),
        compression_level=int(
            config.get("compression_level", _DEFAULT_COMPRESSION_LEVEL)
        ),
        ensure_parent_dir=True,
    )

    print(
        f"Wrote {dh.io.get_n_rows_data_table():,} rows and "
        f"{dh.io.get_n_vars_data_table()} variable(s) to '{output_file!s}'."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
