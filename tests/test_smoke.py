def test_package_imports() -> None:
    import supernode_poc

    assert supernode_poc.__version__ == "0.1.0"
