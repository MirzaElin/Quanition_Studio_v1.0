import importlib

def test_import():
    m = importlib.import_module("quanition_studio")
    assert hasattr(m, "main")
