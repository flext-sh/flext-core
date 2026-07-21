"""Pytest bootstrap for flext-core local package resolution."""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import logging
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Annotated, Literal, TypeVar

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
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
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


class _DocsStub:
    """Permissive placeholder used by documentation snippets."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: object, **kwargs: object) -> _DocsStub:
        return _DocsStub(*args, **kwargs)

    def __getattr__(self, name: str) -> _DocsStub:
        return _DocsStub()

    def __iter__(self) -> Iterator[object]:
        return iter(())

    def __bool__(self) -> bool:
        return True


def _docs_open(file: str | Path, mode: str = "r") -> object:
    """Safe open used in markdown snippets to avoid fixture file coupling."""
    if "b" in mode:
        return Path(file).open(mode)
    try:
        return Path(file).open(mode, encoding="utf-8")
    except FileNotFoundError:
        if "r" in mode:
            return io.StringIO("")
        raise


def _identity_decorator(*_args: object, **_kwargs: object) -> object:
    """Return a decorator that leaves documented functions unchanged."""
    return lambda fn: fn


def pytest_markdown_docs_globals() -> dict[str, object]:
    """Provide shared globals for markdown-docs executable snippets."""
    core = importlib.import_module("flext_core")
    tests_constants = importlib.import_module("tests.constants")
    tests_models = importlib.import_module("tests.models")
    tests_protocols = importlib.import_module("tests.protocols")
    tests_utilities = importlib.import_module("tests.utilities")
    flext_tests = importlib.import_module("flext_tests")

    c = tests_constants.c
    m = tests_models.m
    p = tests_protocols.p
    r = flext_tests.r
    u = tests_utilities.u
    flext_container = core.FlextContainer
    flext_context = core.FlextContext
    flext_settings = core.FlextSettings

    T = TypeVar("T")
    T_co = TypeVar("T_co", covariant=True)
    empty_mapping: dict[str, object] = {}
    stub_names = {
        "User": _DocsStub,
        "Order": _DocsStub,
        "OrderLine": _DocsStub,
        "OrderItem": _DocsStub,
        "OrderStatus": _DocsStub,
        "UserDto": _DocsStub,
        "CreateUserCommand": _DocsStub,
        "GetUserQuery": _DocsStub,
        "UpdateUserCommand": _DocsStub,
        "ProcessOrderCommand": _DocsStub,
        "LongRunningCommand": _DocsStub,
        "SomeCommand": _DocsStub,
        "MyDatabase": _DocsStub,
        "UserRepository": _DocsStub,
        "order_service": _DocsStub(),
        "reserved_emails": set(),
        "users_db": empty_mapping.copy(),
        "large_data": empty_mapping.copy(),
        "data": empty_mapping.copy(),
        "ldif_content": "",
    }

    fence_module = ModuleType("fence")
    fence_module.__dict__.update({
        "c": c,
        "d": getattr(core, "d", _DocsStub),
        "e": getattr(core, "e", _DocsStub),
        "h": getattr(core, "h", _DocsStub),
        "s": getattr(core, "s", _DocsStub),
        "t": getattr(core, "t", _DocsStub),
        "x": getattr(core, "x", _DocsStub),
        "m": getattr(core, "m", m),
        "p": p,
        "r": r,
        "u": u,
        "FlextContainer": flext_container,
        "FlextContext": flext_context,
        "FlextDispatcher": getattr(core, "FlextDispatcher", _DocsStub),
        "FlextLogger": getattr(core, "FlextLogger", _DocsStub),
        "FlextModels": getattr(core, "FlextModels", _DocsStub),
        "FlextSettings": getattr(core, "FlextSettings", flext_settings),
        "Annotated": Annotated,
        "Literal": Literal,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "Never": getattr(core, "Never", _DocsStub),
        "datetime": datetime,
        "timedelta": timedelta,
        "Decimal": Decimal,
        "dataclass": dataclass,
        "json": json,
        "logging": logging,
        "os": os,
        "time": time,
        "T": T,
        "T_co": T_co,
        "ConfigDict": m.ConfigDict,
        "BaseSettings": flext_settings,
        "field_validator": _identity_decorator,
        "model_validator": _identity_decorator,
        "open": _docs_open,
        "complex_calculation": lambda value: value,
        "do_something": lambda: "ok",
        "operation": lambda: "ok",
        "handle_error": lambda _error: None,
        "process_entries": lambda entries: entries,
        **stub_names,
    })
    sys.modules[fence_module.__name__] = fence_module
    return dict(fence_module.__dict__)
