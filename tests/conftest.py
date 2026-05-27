import pytest


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
