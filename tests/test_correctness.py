import auction_assign


def test_package_imports() -> None:
    assert hasattr(auction_assign, "__version__")
