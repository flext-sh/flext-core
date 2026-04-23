"""Comprehensive test configuration and utilities for flext-core.

Provides highly automated testing infrastructure following strict
type-system-architecture.md rules with real functionality testing.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import (
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import pytest

import flext_core as core
from flext_core import FlextContainer, FlextContext, FlextSettings
from tests import c, m, p, r, t, u

collect_ignore_glob = [
    "**/__init__.py",
    "**/_enforcement_integration_fixtures/*.py",
]


@pytest.fixture
def test_context() -> FlextContext:
    """Provide FlextContext instance for testing."""
    return FlextContext()


@pytest.fixture
def clean_container() -> FlextContainer:
    """Provide a clean FlextContainer instance for testing.

    Creates a container and clears all registered services for testing
    in isolation regardless of what other tests may have registered.
    """
    container = FlextContainer()
    container.clear()
    return container


@pytest.fixture
def mock_external_service() -> u.Core.Tests.FunctionalExternalService:
    """Provide mock external service for integration tests."""
    return u.Core.Tests.FunctionalExternalService()


@pytest.fixture
def sample_data() -> dict[
    str, str | int | float | bool | list[str] | dict[str, str] | None
]:
    """Provide sample test data for integration tests."""
    return {
        "string": "test_value",
        "integer": 42,
        "float": math.pi,
        "boolean": True,
        "none": None,
        "list": ["item1", "item2"],
        "dict": {"key": "value"},
    }


@pytest.fixture
def temp_directory(tmp_path: Path) -> Path:
    """Provide temporary directory path for integration tests."""
    return tmp_path


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Temporary directory fixture available to all FLEXT projects."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Temporary file fixture available to all FLEXT projects."""
    return temp_dir / "test_file.txt"


@pytest.fixture
def flext_result_success() -> p.Result[dict[str, bool]]:
    """Successful r fixture available to all FLEXT projects."""
    return r[dict[str, bool]].ok({"success": True})


@pytest.fixture
def flext_result_failure() -> p.Result[str]:
    """Failed r fixture available to all FLEXT projects."""
    return r[str].fail(c.Core.Tests.TestErrors.TEST_ERROR)


@pytest.fixture
def valid_port_numbers() -> Sequence[int]:
    """Valid port numbers for PortNumber validation (1-65535)."""
    return [1, 80, 443, 8080, 3306, 5432, 27017, 65535]


@pytest.fixture
def invalid_port_numbers() -> Sequence[int]:
    """Invalid port numbers for PortNumber validation."""
    return [0, -1, -8080, 65536, 100000]


@pytest.fixture
def valid_uris() -> t.StrSequence:
    """Valid URIs for UriString validation."""
    return [
        "http://localhost",
        "https://example.com",
        "https://example.com:8080",
        "https://example.com/path",
        "https://example.com/path?query=value",
        "https://user:pass@example.com",
        "ftp://files.example.com",
        "grpc://service:50051",
        "postgresql://localhost:5432/db",
        "mongodb://localhost:27017/db",
    ]


@pytest.fixture
def invalid_uris() -> t.StrSequence:
    """Invalid URIs for UriString validation."""
    return [
        "",
        "   ",
        "localhost",
        "example.com",
        "://example.com",
        "http://",
        "http://:8080",
    ]


@pytest.fixture
def valid_hostnames() -> t.StrSequence:
    """Valid hostnames for HostnameStr validation."""
    return [
        "localhost",
        "example.com",
        "sub.example.com",
        "my-server",
        "server-01",
        "api-gateway-v2",
        "db.prod.internal",
        "a",
        "a.b",
    ]


@pytest.fixture
def invalid_hostnames() -> t.StrSequence:
    """Invalid hostnames for HostnameStr validation."""
    return [
        "",
        "   ",
        "-invalid",
        "invalid-",
        "invalid..com",
        "invalid .com",
        "invalid@com",
        "invalid_com",
    ]


@pytest.fixture
def valid_strings() -> t.StrSequence:
    """Valid non-empty strings for string validation."""
    return [
        "a",
        "hello",
        "Hello World",
        "test-value",
        "test_value",
        "test.value",
        "123",
        "value with spaces",
        "UPPERCASE",
        "MixedCase",
    ]


@pytest.fixture
def empty_strings() -> t.StrSequence:
    """Empty strings for validation."""
    return [""]


@pytest.fixture
def whitespace_strings() -> t.StrSequence:
    """Whitespace-only strings for validation."""
    return [" ", "   ", "\t", "\n", "  \t  "]


@pytest.fixture
def valid_ranges() -> Sequence[tuple[int, int, int]]:
    """Valid numeric ranges (value, min, max) for range validation."""
    return [
        (5, 0, 10),
        (0, 0, 10),
        (10, 0, 10),
        (100, 50, 150),
        (-5, -10, 0),
    ]


@pytest.fixture
def invalid_ranges() -> Sequence[tuple[int, int, int]]:
    """Invalid numeric ranges (value, min, max) for range validation."""
    return [
        (-1, 0, 10),
        (11, 0, 10),
        (200, 50, 150),
        (-15, -10, 0),
    ]


@pytest.fixture
def valid_percentages() -> Sequence[float]:
    """Valid percentages (0.0 to 1.0) for percentage validation."""
    return [0.0, 0.5, 0.99, 1.0]


@pytest.fixture
def invalid_percentages() -> Sequence[float]:
    """Invalid percentages for validation."""
    return [-0.1, 1.1, 2.0, -1.0]


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
    if "r" in mode and "b" not in mode:
        try:
            return Path(file).open(mode)
        except FileNotFoundError:
            return io.StringIO("")
    return Path(file).open(mode)


def pytest_markdown_docs_globals() -> dict[str, object]:
    """Provide shared globals for markdown-docs executable snippets."""
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

    return {
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
        "FlextContainer": FlextContainer,
        "FlextContext": FlextContext,
        "FlextDispatcher": getattr(core, "FlextDispatcher", _DocsStub),
        "FlextLogger": getattr(core, "FlextLogger", _DocsStub),
        "FlextModels": getattr(core, "FlextModels", _DocsStub),
        "FlextSettings": getattr(core, "FlextSettings", FlextSettings),
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
        "BaseSettings": FlextSettings,
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
    }
