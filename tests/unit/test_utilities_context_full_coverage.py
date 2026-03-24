"""Tests for FlextUtilitiesContext - proxy creation and clone utilities.

Module: flext_core._utilities.context
Coverage target: lines 139-158, 180

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from flext_tests import t
from pydantic import BaseModel

from flext_core import FlextContext, p
from tests import u

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
            self.runtime_dispatcher = None
            self.runtime_registry = None
            self.runtime_context = None
            self.runtime_config = None

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

    def test_clone_runtime_copies_dispatcher(self) -> None:
        """Cloned runtime has the same runtime_dispatcher."""
        runtime = self._FakeRuntime()
        cloned = u.clone_runtime(runtime)
        assert cloned.runtime_dispatcher is runtime.runtime_dispatcher

    def test_clone_runtime_copies_registry(self) -> None:
        """Cloned runtime has the same runtime_registry."""
        runtime = self._FakeRuntime()
        cloned = u.clone_runtime(runtime)
        assert cloned.runtime_registry is runtime.runtime_registry

    def test_clone_runtime_uses_provided_context(self) -> None:
        """When context is provided, cloned runtime uses it."""
        runtime = self._FakeRuntime()
        new_context = FlextContext.create()
        cloned = u.clone_runtime(runtime, context=new_context)
        assert cloned.runtime_context is new_context

    def test_clone_runtime_copies_context_when_not_provided(self) -> None:
        """When context is None, cloned runtime copies original context."""
        runtime = self._FakeRuntime()
        cloned = u.clone_runtime(runtime)
        assert cloned.runtime_context is runtime.runtime_context

    def test_clone_runtime_applies_config_overrides(self) -> None:
        """When config_overrides provided, model_copy is called with them."""
        runtime = self._FakeRuntime()
        cloned = u.clone_runtime(
            runtime, config_overrides=t.ConfigMap(root={"timeout": 30})
        )
        assert isinstance(cloned.runtime_config, TestUnitModels._FakeConfig)
        assert cloned.runtime_config.data["timeout"] == 30

    def test_clone_runtime_copies_config_when_no_overrides(self) -> None:
        """When no config_overrides, config is copied as-is."""
        runtime = self._FakeRuntime()
        cloned = u.clone_runtime(runtime)
        assert cloned.runtime_config is runtime.runtime_config

    def test_clone_runtime_handles_missing_attributes(self) -> None:
        """Runtime with None attributes still clones successfully."""
        runtime = self._MinimalRuntime()
        cloned = u.clone_runtime(runtime)
        assert isinstance(cloned, self._MinimalRuntime)
        assert cloned.runtime_dispatcher is None
        assert cloned.runtime_config is None

    def test_clone_container_calls_scoped(self) -> None:
        """Calls container.scoped() with provided args."""
        container = MagicMock()
        expected = MagicMock()
        container.scoped.return_value = expected
        result = u.clone_container(
            container, scope_id="test-scope", overrides={"service": "mock"}
        )
        container.scoped.assert_called_once_with(
            subproject="test-scope",
            services={"service": "mock"},
        )
        assert result is expected

    def test_clone_container_with_defaults(self) -> None:
        """With no args, scoped() is called with None defaults."""
        container = MagicMock()
        u.clone_container(container)
        container.scoped.assert_called_once_with(subproject=None, services=None)
