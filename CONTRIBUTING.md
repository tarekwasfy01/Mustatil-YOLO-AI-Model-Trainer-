# Contributing to Mustatil

Thank you for considering a contribution.

## Before starting

1. Search existing issues to avoid duplication.
2. Open an issue for substantial changes before writing code.
3. Keep changes focused and explain the research or user need they address.

## Development setup

The repository should contain the unpacked Python source and its packaging metadata at the repository root. Create a Python 3.10–3.12 environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

For headless Qt testing on Linux:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

## Pull requests

- Add or update tests for changed behaviour.
- Update user documentation when behaviour changes.
- Add an entry to `CHANGELOG.md` under **Unreleased**.
- Do not commit models, imagery, credentials, or datasets unless their licence and size make repository inclusion appropriate.
- Describe platform, Python version, and hardware used for testing.

## Reporting scientific or geospatial errors

When reporting coordinate, projection, tiling, or export problems, include a minimal non-sensitive input, its coordinate reference system, expected result, actual result, and the exact Mustatil version.
