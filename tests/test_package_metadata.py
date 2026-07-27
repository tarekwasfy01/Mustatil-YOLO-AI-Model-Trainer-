from importlib.metadata import PackageNotFoundError, version


def test_installed_distribution_has_version() -> None:
    try:
        installed = version("mustatil")
    except PackageNotFoundError as exc:
        raise AssertionError("Install the project with `python -m pip install -e .` before testing.") from exc
    assert installed.strip(), "The installed package version must not be empty."
