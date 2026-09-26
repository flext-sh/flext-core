"""Pytest bootstrap for flext-core local package resolution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent


def _install_local_package(package_name: str, package_dir: Path) -> None:
    init_file = package_dir / "__init__.py"
    existing_package = sys.modules.get(package_name)
    if (
        existing_package is not None
        and Path(getattr(existing_package, "__file__", "")).resolve() == init_file
    ):
        return

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    package_spec = importlib.util.spec_from_file_location(
        package_name, init_file, submodule_search_locations=[str(package_dir)]
    )
    if package_spec is None or package_spec.loader is None:
        msg = f"Unable to load local package from {init_file}"
        raise ImportError(msg)

    package_module = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package_module
    package_spec.loader.exec_module(package_module)


for local_package in ("examples", "tests"):
    local_dir = _PROJECT_ROOT / local_package
    if local_dir.is_dir() and (local_dir / "__init__.py").is_file():
        _install_local_package(local_package, local_dir)
