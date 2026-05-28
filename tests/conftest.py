import pytest


def pytest_configure(config: pytest.Config) -> None:
    config._pplx_garden_has_native_extension = True
    try:
        import torch  # noqa: F401

        __import__("pplx_garden._rust")
    except ImportError as exc:
        config._pplx_garden_has_native_extension = False
        config._pplx_garden_native_extension_error = exc


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-multinode",
        action="store_true",
        default=False,
        help="run tests that require multiple launched nodes",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-multinode"):
        return

    skip_multinode = pytest.mark.skip(
        reason="requires --run-multinode and one pytest process per node",
    )
    for item in items:
        if "multinode" in item.keywords:
            item.add_marker(skip_multinode)

    if getattr(config, "_pplx_garden_has_native_extension", False):
        return

    reason = "pplx_garden native extension is not importable"
    error = getattr(config, "_pplx_garden_native_extension_error", None)
    if error is not None:
        reason = f"{reason}: {error!r}"
    skip_native = pytest.mark.skip(reason=reason)
    for item in items:
        if "fabric" in item.keywords or "kernel" in item.keywords:
            item.add_marker(skip_native)
