"""Verifica que el paquete y sus submódulos cargan sin errores."""

import importlib


def test_package_imports():
    pkg = importlib.import_module("exoplanet")
    assert pkg.__version__ == "0.1.0"


def test_subpackages_import():
    for sub in ("data", "models", "training", "evaluation", "utils"):
        importlib.import_module(f"exoplanet.{sub}")
