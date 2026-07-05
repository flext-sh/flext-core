"""Comprehensive test configuration and utilities for flext-core.

Provides highly automated testing infrastructure following strict
type-system-architecture.md rules with real functionality testing.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import os
import sys
import time
from collections.abc import (
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Annotated, Literal, TypeVar, cast

import pytest
from flext_tests import (
    clean_container as _shared_clean_container,
    r,
    reset_settings as _shared_reset_settings,
    sample_data as _shared_sample_data,
    settings as _shared_settings,
    settings_factory as _shared_settings_factory,
    temp_dir as _shared_temp_dir,
    temp_file as _shared_temp_file,
    test_context as _shared_test_context,
    test_runtime as _shared_test_runtime,
)

from tests.constants import c
from tests.models import m
from tests.protocols import p
from tests.utilities import u

clean_container = _shared_clean_container
reset_settings = _shared_reset_settings
sample_data = _shared_sample_data
settings = _shared_settings
settings_factory = _shared_settings_factory
temp_dir = _shared_temp_dir
temp_file = _shared_temp_file
test_context = _shared_test_context
test_runtime = _shared_test_runtime

collect_ignore_glob = [
    "**/__init__.py",
]


@pytest.fixture
def mock_external_service() -> u.Tests.FunctionalExternalService:
    """Provide mock external service for integration tests."""
    return u.Tests.FunctionalExternalService()


class _DocsStub:
    """Permissive placeholder used by documentation snippets.

    It allows docs examples to reference illustrative names without failing on
    attribute access/calls when the snippet is intentionally partial.
    """

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


def pytest_markdown_docs_globals() -> dict[str, object]:
    """Provide shared globals for markdown-docs executable snippets."""
    core = importlib.import_module("flext_core")
    flext_container = core.FlextContainer
    flext_context = core.FlextContext
    flext_settings = core.FlextSettings

    T = TypeVar("T")
    T_co = TypeVar("T_co", covariant=True)
    empty_mapping: dict[str, object] = {}

    # Common placeholders used by docs snippets as illustrative identifiers.
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

    runtime_module = ModuleType("tests._markdown_docs_runtime")
    sys.modules[runtime_module.__name__] = runtime_module
    runtime_module.__dict__.update({
        # Canonical runtime aliases and common classes.
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
        # Typing / stdlib helpers commonly referenced in snippets.
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
        # Pydantic v2 symbols frequently referenced by docs.
        "ConfigDict": m.ConfigDict,
        "BaseSettings": flext_settings,
        "field_validator": lambda *args, **kwargs: lambda fn: fn,
        "model_validator": lambda *args, **kwargs: lambda fn: fn,
        # Common helpers used by snippet examples.
        "open": _docs_open,
        "complex_calculation": lambda value: value,
        "do_something": lambda: "ok",
        "operation": lambda: "ok",
        "handle_error": lambda _error: None,
        "process_entries": lambda entries: entries,
        **stub_names,
    })
    return cast("dict[str, object]", runtime_module.__dict__)
