# Oscana: Neutrino Oscillation Analysis Package

## Project Installation

The project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
pip install poetry
```

After cloning the repository, load the dependencies into a virtual environment.

```bash
poetry install
```

## Loading files

Oscana currently supports: MINOS SNTP files and HDF5 files.

**Step 1**: Import and initialise Oscana.

```python
import oscana

oscana.init()
```

This adds the variables from the ".env" file to environment variables. The ".env" could be used to store the directories to the files; note that it should also work by simply passing the directory!

It will also set up logging - you can specify where you would like the log files to be saved.

**Step 2**: Create the data handler.

Define your variables.

```python
# Option 1: Uproot-friendly names.
variables = [
    "NtpSt/stp.planeview",
    "NtpSt/stp.strip",
    "NtpSt/stp.plane",
]

# Option 2: Use a `VariableCollection` - "constants.py".
variables = oscana.IMAGE_BASIC_VARIABLES.uproot
```

Initialise the data handler. This will use the default data strategy (Pandas).

```python
dh = oscana.data.DataHandler(
    variables=variables,
    make_cut_bool_table=False,
)
```

Enabling the "cut bool" table means that cuts are stored separately as boolean arrays rather than being applied directly onto the loaded data.

**Step 3**: Load the files.

If you want to load it from ROOT SNTP files.

```python
dh.io.from_sntp(files=[...])
```

If you want to load it from HDF5 files.

```python
dh.io.from_hdf5(files=[...])
```

Note that you could either specify the path or a variable in the ".env" file which contains the absolute path. For example, you could have `MY_SNTP="/home/someone/data/my_file.root"` in the ".env" file.

*Important*: The files that you are trying to load must have the same metadata as the files that have already been loaded. This is to prevent, for example, loading both forward and reverse horn current data at the same time.

You can access both tables by doing `dh.data` or `dh.cuts`. If the cuts table is disabled it will throw an error.

## Saving as HDF5 files

To save the loaded table and its metadata to HDF5:

```python
dh.io.to_hdf5(
    out_file="./out.h5",
    # Optionally, compress the output file.
    compression="gzip",
    compression_level=6,
)
```
