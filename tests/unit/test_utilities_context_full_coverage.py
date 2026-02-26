"""Tests for FlextUtilitiesContext - proxy creation and clone utilities.

Module: flext_core._utilities.context
Coverage target: lines 139-158, 180

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock

from flext_core import p, t, u
from pydantic import BaseModel


class TestCreateStrProxy:
    """Tests for u.Context.create_str_proxy()."""

    def test_create_str_proxy_basic(self) -> None:
        """Creates a StructlogProxyContextVar[str] with given key."""
        proxy = u.Context.create_str_proxy("my_key")
        assert proxy is not None
        assert proxy._key == "my_key"

    def test_create_str_proxy_with_default(self) -> None:
        """Creates proxy with default value."""
        proxy = u.Context.create_str_proxy("my_key", default="fallback")
        assert proxy._default == "fallback"


class TestCreateDatetimeProxy:
    """Tests for u.Context.create_datetime_proxy()."""

    def test_create_datetime_proxy_basic(self) -> None:
        """Creates a StructlogProxyContextVar[datetime] with given key."""
        proxy = u.Context.create_datetime_proxy("start_time")
        assert proxy is not None
        assert proxy._key == "start_time"

    def test_create_datetime_proxy_with_default(self) -> None:
        """Creates proxy with datetime default."""
        now = datetime.now(UTC)
        proxy = u.Context.create_datetime_proxy("start_time", default=now)
        assert proxy._default == now


class TestCreateDictProxy:
    """Tests for u.Context.create_dict_proxy()."""

    def test_create_dict_proxy_basic(self) -> None:
        """Creates a StructlogProxyContextVar[dict] with given key."""
        proxy = u.Context.create_dict_proxy("metadata")
        assert proxy is not None
        assert proxy._key == "metadata"

    def test_create_dict_proxy_with_default(self) -> None:
        """Creates proxy with dict default."""
        default_val = t.ConfigMap(root={"key": "value"})
        proxy = u.Context.create_dict_proxy("metadata", default=default_val)
        assert proxy._default == default_val


class _FakeConfig(BaseModel):
    """Fake config with model_copy support."""

    timeout: int = 10

    @property
    def data(self) -> dict[str, object]:
        return {"timeout": self.timeout}


class _FakeRuntime:
    """Fake runtime class with all expected attributes."""

    def __init__(self) -> None:
        self._dispatcher = object()
        self._registry = object()
        self._context = object()
        self._config = _FakeConfig(timeout=10)


class _FakeContext:
    def clone(self) -> _FakeContext:
        return _FakeContext()

    def set(self, key: str, value: t.GeneralValueType) -> None:
        _ = key, value

    def get(self, key: str) -> object:
        _ = key
        return None


class TestCloneRuntime:
    """Tests for u.Context.clone_runtime()."""

    def test_clone_runtime_copies_dispatcher(self) -> None:
        """Cloned runtime has the same _dispatcher."""
        runtime = _FakeRuntime()
        cloned = u.Context.clone_runtime(runtime)
        assert cloned._dispatcher is runtime._dispatcher

    def test_clone_runtime_copies_registry(self) -> None:
        """Cloned runtime has the same _registry."""
        runtime = _FakeRuntime()
        cloned = u.Context.clone_runtime(runtime)
        assert cloned._registry is runtime._registry

    def test_clone_runtime_uses_provided_context(self) -> None:
        """When context is provided, cloned runtime uses it."""
        runtime = _FakeRuntime()
        new_context = _FakeContext()
        cloned = u.Context.clone_runtime(
            runtime,
            context=cast("p.Context", cast("object", new_context)),
        )
        assert cloned._context is new_context

    def test_clone_runtime_copies_context_when_not_provided(self) -> None:
        """When context is None, cloned runtime copies original context."""
        runtime = _FakeRuntime()
        cloned = u.Context.clone_runtime(runtime)
        assert cloned._context is runtime._context

    def test_clone_runtime_applies_config_overrides(self) -> None:
        """When config_overrides provided, model_copy is called with them."""
        runtime = _FakeRuntime()
        cloned = u.Context.clone_runtime(
            runtime,
            config_overrides=t.ConfigMap(root={"timeout": 30}),
        )
        assert isinstance(cloned._config, _FakeConfig)
        assert cloned._config.data["timeout"] == 30

    def test_clone_runtime_copies_config_when_no_overrides(self) -> None:
        """When no config_overrides, config is copied as-is."""
        runtime = _FakeRuntime()
        cloned = u.Context.clone_runtime(runtime)
        assert cloned._config is runtime._config

    def test_clone_runtime_handles_missing_attributes(self) -> None:
        """Runtime without optional attributes still clones successfully."""

        class MinimalRuntime:
            pass

        runtime = MinimalRuntime()
        cloned = u.Context.clone_runtime(runtime)
        assert isinstance(cloned, MinimalRuntime)


class TestCloneContainer:
    """Tests for u.Context.clone_container()."""

    def test_clone_container_calls_scoped(self) -> None:
        """Calls container.scoped() with provided args."""
        container = MagicMock()
        expected = MagicMock()
        container.scoped.return_value = expected

        result = u.Context.clone_container(
            container,
            scope_id="test-scope",
            overrides={"service": "mock"},
        )
        container.scoped.assert_called_once_with(
            subproject="test-scope",
            services={"service": "mock"},
        )
        assert result is expected

    def test_clone_container_with_defaults(self) -> None:
        """With no args, scoped() is called with None defaults."""
        container = MagicMock()
        u.Context.clone_container(container)
        container.scoped.assert_called_once_with(
            subproject=None,
            services=None,
        )
