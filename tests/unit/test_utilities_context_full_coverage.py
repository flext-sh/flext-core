"""Tests for FlextUtilitiesContext - proxy creation and clone utilities.

Module: flext_core._utilities.context
Coverage target: lines 139-158, 180

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from pydantic import BaseModel

from tests import p, t, u

from ._models import TestUnitModels


class TestUtilitiesContextFullCoverage:
    class _FakeRuntime:
        """Fake runtime class implementing p.CloneableRuntime protocol."""

        def __init__(self) -> None:
            self.runtime_dispatcher: p.Dispatcher | None = MagicMock(spec=p.Dispatcher)
            self.runtime_registry: p.Registry | None = MagicMock(spec=p.Registry)
            self.runtime_context: p.Context | None = MagicMock(spec=p.Context)
            self.runtime_config: BaseModel | None = TestUnitModels._FakeConfig(
                timeout=10,
            )

    class _FakeContext:
        def clone(self) -> TestUtilitiesContextFullCoverage._FakeContext:
            return TestUtilitiesContextFullCoverage._FakeContext()

        def set(self, key: str, value: t.Scalar) -> None:
            _ = (key, value)

        def get(self, key: str) -> None:
            _ = key

    class _MinimalRuntime:
        """Minimal runtime with None defaults for all protocol attributes."""

        def __init__(self) -> None:
            self.runtime_dispatcher: p.Dispatcher | None = None
            self.runtime_registry: p.Registry | None = None
            self.runtime_context: p.Context | None = None
            self.runtime_config: BaseModel | None = None

    def test_create_str_proxy_basic(self) -> None:
        """Creates a StructlogProxyContextVar[str] with given key."""
        proxy = u.create_str_proxy("my_key")
        assert proxy is not None
        assert proxy._key == "my_key"

    def test_create_str_proxy_with_default(self) -> None:
        """Creates proxy with default value."""
        proxy = u.create_str_proxy("my_key", default="fallback")
        assert proxy._default == "fallback"

    def test_create_datetime_proxy_basic(self) -> None:
        """Creates a StructlogProxyContextVar[datetime] with given key."""
        proxy = u.create_datetime_proxy("start_time")
        assert proxy is not None
        assert proxy._key == "start_time"

    def test_create_datetime_proxy_with_default(self) -> None:
        """Creates proxy with datetime default."""
        now = datetime.now(UTC)
        proxy = u.create_datetime_proxy("start_time", now)
        assert proxy._default == now

    def test_create_dict_proxy_basic(self) -> None:
        """Creates a StructlogProxyContextVar[dict] with given key."""
        proxy = u.create_dict_proxy("metadata")
        assert proxy is not None
        assert proxy._key == "metadata"

    def test_create_dict_proxy_with_default(self) -> None:
        """Creates proxy with dict default."""
        default_val = t.ConfigMap(root={"key": "value"})
        proxy = u.create_dict_proxy("metadata", default_val)
        assert proxy._default == default_val
