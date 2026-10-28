"""Behavioral tests for the FlextContainer facade lifecycle contract.

Complements the split registration/resolution suites by exercising the
container's facade-level public behaviors: singleton identity, drop, clear,
scope isolation, fluent apply, and snapshot. Every assertion targets the
observable public surface (return values, ``r[T]`` outcomes, and public
model state) rather than internal structures.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect

import pytest

from flext_core import FlextContainer
from tests import m, p, t, tm


class TestsFlextCoreContainer:
    """Tests for ``FlextCoreContainer``."""

    @staticmethod
    def test_singleton_returns_same_instance() -> None:
        """Constructing the container twice yields the same shared instance."""
        first = FlextContainer()
        second = FlextContainer()
        tm.that(
            first is second,
            eq=True,
            msg="FlextContainer() must return the process-wide singleton",
        )

    @staticmethod
    def test_reset_for_testing_replaces_singleton() -> None:
        """reset_for_testing drops the cached singleton so a fresh one is built."""
        original = FlextContainer()
        FlextContainer.reset_for_testing()
        replacement = FlextContainer()
        tm.that(
            replacement is not original,
            eq=True,
            msg="reset_for_testing must force a brand-new singleton instance",
        )

    @staticmethod
    def test_logger_requires_explicit_module_name(
        clean_container: p.Container,
    ) -> None:
        """Test logger requires explicit module name."""
        params = inspect.signature(type(clean_container).logger).parameters
        module_param = params["module_name"]
        tm.that(
            module_param.default is inspect.Parameter.empty,
            eq=True,
            msg="module_name must be required, not optional",
        )

    @staticmethod
    def test_clear_is_idempotent(clean_container: p.Container) -> None:
        """Clearing an already-empty container remains empty (no error)."""
        clean_container.clear()
        clean_container.clear()
        tm.that(
            clean_container.names(),
            empty=True,
            msg="Repeated clear must keep the container empty",
        )

    @staticmethod
    def test_scope_preserves_explicit_settings_provider(
        clean_container: p.Container,
    ) -> None:
        """An injected provider keeps its identity without changing the parent."""
        parent_settings = clean_container.settings
        injected = parent_settings.clone()

        scoped = clean_container.scope(
            registration=m.ServiceRegistrationSpec(settings=injected),
        )

        tm.ok(scoped.settings is injected)
        tm.ok(clean_container.settings is parent_settings)
        tm.ok(scoped.settings is not parent_settings)

    @staticmethod
    @pytest.mark.parametrize(
        "overrides",
        [None, {"custom_key": "custom_value"}],
        ids=["none", "mapping"],
    )
    def test_apply_returns_self_for_fluent_chaining(
        overrides: t.UserOverridesMapping | None,
        clean_container: p.Container,
    ) -> None:
        """Apply returns the same container so callers can chain fluently."""
        tm.that(
            clean_container.apply(overrides) is clean_container,
            eq=True,
            msg="apply must return self for fluent configuration",
        )

    @staticmethod
    def test_snapshot_exposes_config_via_model_dump(
        clean_container: p.Container,
    ) -> None:
        """Snapshot exposes merged settings as a mapping via its public model API."""
        dumped = clean_container.snapshot().model_dump()
        tm.that(
            dumped,
            is_=dict,
            msg="snapshot().model_dump() must return a plain mapping of settings",
        )
