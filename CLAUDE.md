# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Oscana is a single-author Python package for neutrino oscillation analysis (MINOS in
practice): it reads MINOS SNTP ROOT files with `uproot`, applies an ordered chain of
cuts/transforms, and writes the tables **plus full provenance metadata** to HDF5 for data
preservation. It also provides MINOS-specific plotting and far-detector event imaging.
Poetry-managed, `src/oscana` layout, Python >=3.10,<3.12.

## Commands

```bash
poetry install
poetry run python -m pytest                 # tests (pythonpath=["src"]); tests/ is gitignored, may be absent
poetry run python -m pytest tests/test_x.py::test_name
poetry run black --line-length 80 .         # also format-on-save in VSCode
poetry run mypy src/oscana                  # mypy.ini pins .venv/bin/python
poetry run pylint src/oscana                # .pylintrc, max-line-length 80
poetry build                                # -> dist/

# ROOT/HDF5 -> HDF5 from a JSON config; --help prints an annotated example config
poetry run python scripts/build_h5.py <config.json> <out.h5> [--overwrite] \
    [--logs-dir DIR] [--verbosity DEBUG|INFO|WARNING|ERROR|CRITICAL] [--dotenv-dir DIR]

poetry run python scripts/get_variables.py  # regenerates res/variables/*.txt from a ROOT file
```

## Runtime facts that code depends on

- **`oscana.init()` must run before anything else** (logger + `.env` + `res/numbers.json`
  into the `minos_numbers` dict). Importing alone leaves logging unconfigured and
  `minos_numbers` empty. Sub-steps: `init_root_logger`, `init_env_variables`,
  `init_minos_numbers`. Importing `oscana` also prints a version banner.
- **`.env` is a file-name registry**, not just config: keys like
  `SNTP_F21042000_0024_D5_D07_R7` / `uDST_2010_MC_FD_R1_1` map short code names to
  absolute paths. `_resolve_file_directory` tries the env lookup first, then falls back
  to treating the string as a path — so `files` entries are usually these code names.
  Gitignored.
- **WSL rewriting**: `_apply_wsl_prefix` turns `C://...` into `/mnt/c/...` on Linux so one
  `.env` works from Windows and WSL. Deliberately duplicated in `logger.py` (circular
  import) and `scripts/get_variables.py` (copy-paste).
- `res/` ships with the wheel, found via `RESOURCES_PATH` (`constants.py`): `numbers.json`
  (MINOS detector geometry/masses), `colours.json`, `configs/logging.json`, `fonts/`,
  `variables/*.txt` (dumps of valid SNTP variable names).

## Module map

| Path | Contents |
| --- | --- |
| `constants.py` | `RESOURCES_PATH`, `IMAGE_DTYPE`/`NUM_DTYPE` (float32), SNTP branch names (`NtpSt`, `NtpBDLite`, `NtpFitSA`), `VariableCollection` + named collections, physics enums (`EIAction`, `EIdHEP`, `EInteraction`, `EPlaneView`, …) |
| `utils.py` | `init_env_variables`, `init_minos_numbers`, `minos_numbers`, `_get_dir_from_env`, `_apply_wsl_prefix`, `import_plugins`, `get_func_lookup`, `get_bin_centers`, `VariableSearchTool` |
| `logger.py` | `init_root_logger`, `_error`, `_warn` |
| `errors.py` / `escape.py` | `OscanaError`; ANSI `Style` codes (internal, `__all__ = []`) |
| `themes.py` | `Theme` dataclass + `themes` dict — keys are **lowercase** (`slate`, `sandyslate`, `light`, `draft`, `nova`, `nova-nu26-darkblue/-lightblue`); an unknown name warns and falls back to `draft` |
| `plotting.py` | `plotting_context`; layouts (`grid_layout`, `spectrum_layout`, `fd_uv_views_layout`, `marginal_hist_layout`), modifiers (`energy_axs_scale`, `spec_fig_cleanup`, `add_experiment_tag`), plots (`plot_hist`, `plot_hist_from_heights`) and templates (`plot_energy_resolution`, `plot_fd_event_image`, `plot_2d_hist`) |
| `images.py` | FD event images from strip/plane hits: `create_fd_full_image`, `create_fd_split_image`, `create_fd_crop_image`, `get_image_profiles`, `image_to_sparse` — geometry comes from `minos_numbers` |
| `data/` | `DataHandler`, `_DataIOStrategy`, `TransformBase`, `DataCallbackBase`, `FileMetadata`, `TransformMetadata`, file enums, `plugins/` |
| `scripts/` | `build_h5.py` (config-driven HDF5 builder), `get_variables.py` |

## Architecture

### DataHandler + IO strategy plugins

`DataHandler` ([data_handler.py](src/oscana/data/data_handler.py)) is `Generic[T]` over the
table type and owns a **data table**, an optional **cuts boolean table**
(`make_cut_bool_table=True`), a list of `FileMetadata`, and one `TransformMetadata`. It has
no IO logic — that sits behind `_DataIOStrategy` ([io_base.py](src/oscana/data/io_base.py)),
reached as `dh.io`. Tables are **locked by default**; transforms unlock/mutate/re-lock.

Plugins are discovered dynamically: `utils.import_plugins(file=__file__)` globs
`data/plugins/*.py`, imports each, and registers every name in its `__all__`;
`DataHandler(variables=[...], data_io="PandasIO")` resolves the class by that string.
**Adding a backend = drop a module in `plugins/` exporting the class in `__all__`** — no
other registration. `PandasIO` ([pandas_io.py](src/oscana/data/plugins/pandas_io.py)) is
the only one today (`from_sntp`, `from_udst`, `from_hdf5`, `to_hdf5`, `get_n_rows_*`,
`get_vars_*`).

Loading (`_load_from_files`) is batched and fault-tolerant on purpose: per-file frames are
built as "mini DataFrames", per-file exceptions are collected rather than raised, and only
`_update_parent` merges at the end — one bad file will not kill an interactive session.
A `_cache` of *resolved absolute paths* prevents double-loading the same file under two
code names. `_post_file_loader_checks` rejects a file whose columns, cut columns, or
transform history disagree with what is already loaded, or that carries cuts into a
handler without a cuts table.

### Transforms, cuts, callbacks

`TransformBase` ([transform.py](src/oscana/data/transform.py)) subclasses are callables
`(dh) -> (data_table, cuts_table)`; `__init__(**kwargs)` stashes kwargs for metadata.
**The class name is load-bearing**: `^(tfm|cut)_(\d{8})_(.+)$`, e.g.
`cut_20260325_fiducial_volume` — the prefix decides cut vs transform, the date is the
version. `dh.apply_transforms(transforms, callbacks=...)` applies in order, records each in
`TransformMetadata`, fires `DataCallbackBase.before_transform`/`after_transform`, and (like
loading) warns per failure and raises once at the end.

`utils.get_func_lookup(globals_, prefix)` returns a lookup that sorts matching names and
hands back the **latest-dated** one, so callers ask for `"fiducial_volume"` and get the
newest revision. Transform modules used by `build_h5.py` must define `cut_lookup` and
`transform_lookup` at the bottom via this helper.

### Variables

MINOS ROOT names are long paths; `VariableCollection` ([constants.py](src/oscana/constants.py))
groups them under a branch and exposes `.list_` (bare) / `.uproot` (branch-prefixed), and
behaves like a list (`+`, `len`, indexing, iteration). Exported collections:
`HEADER_VARIABLES`, `IMAGE_{BASIC,PE,SIGCOR,TIME,ALL}_VARIABLES`, `EVENT_VERTEX_VARIABLES`,
`MC_4MOMENTUM_VARIABLES`, `MC_INTERACTION_VARIABLES` — all rooted at `NtpSt`.
`build_h5.py` resolves each config `variables` entry as either a collection name exported
by `oscana` or a literal `"<branch>/<variable>"`. `VariableSearchTool` (Borg singleton in
`utils.py`) searches the `res/variables/*.txt` dumps: `init_lookup_table`,
`search_for_variable`, `print_roots`, `print_variables`.

### HDF5 format

Three top-level groups (`H5_*_BRANCH_NAME` in `io_base.py`): `data`, `cuts`, `meta`.
One dataset per column; jagged (object/ndarray) columns are written as `h5py` vlen types,
and an optional `data_type_converter(column, dtype) -> dtype` can override per column.
Compression is `gzip`/`lzf`/`None` (no szip — licensing). `meta` holds three JSON-encoded
UTF-8 string datasets: `variables`, `transforms` (`TransformMetadata.to_dict`, keys
uniquified as `uid_{i:09d}_{ClassName}`), `files` (list of `FileMetadata.to_dict`).
Metadata round-trip fidelity is the point of the format — reloading restores the transform
history, which is then enforced across merges. Output must end `.h5`.

### File metadata

`FileMetadata` ([f_metadata.py](src/oscana/data/f_metadata.py)) is a frozen slots dataclass
built by `from_sntp`, which currently only understands **Daikon MC spill file names**
(`_daikon_regex`: detector / interaction region / flavour / B-field / run / subrun / horn
position / target Z shift / current + sign / MC vegetable + version / beam-flux run /
file type / reco wood + version) and errors on anything else. Start/end times and record
counts come from the `NtpSt` header branch (UTC seconds → `datetime`); the run number is
read per-event and reduced by mode if a file spans runs. Every field is enum-typed
([enumerations.py](src/oscana/data/enumerations.py)) and every enum defines `_missing_`, so
unknown values degrade to an `UNKNOWN` sentinel instead of raising.

### Logging and errors

Everything logs to `logging.getLogger("Root")`, configured from `res/configs/logging.json`
(stdout at the chosen verbosity, `FullLog.log` at DEBUG, `Errors.log` at ERROR, both
rotating). Use `_error(ExcType, msg, logger)` (logs then raises) and
`_warn(WarnType, msg, logger)`. **Do not use `_error` for exceptions the caller will
catch** — it pollutes `Errors.log`; plain `raise` there (see `utils._get_dir_from_env`).

## Conventions

- 80 columns everywhere (Black, pylint, VSCode ruler).
- `from __future__ import annotations` at the top; explicit `__all__` in every module
  (`[]` for internal-only ones) since packages re-export via `from .x import *`.
- Numpy-style docstrings opened as `"""\` so the summary starts on the next line;
  `# ===== [ Section ] ===== #` banners padded to 80 cols.
- `_`-prefixed names are internal; their docstrings start with `[ Internal ]`.
- Version lives in both `pyproject.toml` and `__init__.__version__` — keep in sync
  (semver since 30/10/2025).
- British spelling in identifiers (`colour`, `initialise`).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
