# tests/test_smoke.py
import importlib
from pathlib import Path


def test_dataset_present():
    assert Path("data/Bluno_TradeFinanceAgent_Dataset_v1.xlsx").exists()


def test_app_package_imports():
    assert importlib.import_module("app") is not None
