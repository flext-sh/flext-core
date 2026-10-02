"""Behavioral tests for the truthful container write rules.

Every registration passes one write path: empty, duplicate and reserved names
raise ``e.ValidationError``; the reserved core services stay private to the
container, including in scopes; factory auto-registration never skips.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import pytest
from flext_tests import d, tm

from flext_core.container import FlextContainer
from tests import e
from tests.constants import c
from tests.models import m
from tests.protocols import p
from tests.utilities import u

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextCoreContainerTruthful:
    """Exercise the rules of the single container write path."""

    @staticmethod
    def _shared_with_auto_registration() -> None:
        _ = FlextContainer.shared(auto_register_factories=True)

    @pytest.mark.parametrize("name", sorted(c.CONTAINER_RESERVED_NAMES))
    def test_public_writes_reject_reserved_names(
        self,
        name: str,
        clean_container: p.Container,
    ) -> None:
        """bind, factory and resource refuse every core runtime name."""
        factory = u.Tests.create_factory("value")
        for write in (
            lambda: clean_container.bind(name, "value"),
            lambda: clean_container.factory(name, factory),
            lambda: clean_container.resource(name, factory),
        ):
            with pytest.raises(e.ValidationError, match="reserved"):
                _ = write()
        tm.ok(clean_container.resolve(name))
        tm.that(clean_container.has(name), eq=False)

    def test_duplicate_is_rejected_across_registration_kinds(
        self,
        clean_container: p.Container,
    ) -> None:
        """A name bound as a service cannot be reused by a factory or resource."""
        factory = u.Tests.create_factory("other")
        _ = clean_container.bind("shared_name", "service")

        with pytest.raises(e.ValidationError, match="shared_name"):
            _ = clean_container.factory("shared_name", factory)
        with pytest.raises(e.ValidationError, match="shared_name"):
            _ = clean_container.resource("shared_name", factory)

        tm.ok(clean_container.resolve("shared_name"), eq="service")

    def test_resource_empty_name_raises(self, clean_container: p.Container) -> None:
        """An empty resource name raises instead of being ignored."""
        with pytest.raises(e.ValidationError, match=c.ERR_CONTAINER_NAME_EMPTY):
            _ = clean_container.resource("", u.Tests.create_factory("value"))

    def test_drop_of_reserved_name_fails_and_keeps_core_service(
        self,
        clean_container: p.Container,
    ) -> None:
        """Core services are not public, so drop reports them as not found."""
        tm.fail(clean_container.drop(c.ServiceName.LOGGER), has="logger")
        tm.ok(clean_container.resolve(c.ServiceName.LOGGER))

    def test_scope_keeps_logger_internal(self, clean_container: p.Container) -> None:
        """Regression: a scope re-registers LOGGER as internal, never public."""
        _ = clean_container.bind("public", "value")

        scoped = clean_container.scope()
        nested = scoped.scope()

        for container in (scoped, nested):
            tm.that(list(container.names()), eq=["public"])
            tm.that(container.has(c.ServiceName.LOGGER), eq=False)
            logger = tm.ok(container.resolve(c.ServiceName.LOGGER))
            tm.that(isinstance(logger, p.Logger), eq=True)
            with pytest.raises(e.ValidationError, match="reserved"):
                _ = container.factory(
                    c.ServiceName.LOGGER,
                    u.Tests.create_factory("value"),
                )

    def test_scope_binds_core_services_to_its_own_runtime(
        self,
        clean_container: p.Container,
    ) -> None:
        """The scoped settings and context services are the scope's own."""
        scoped = clean_container.scope(subproject="unit")

        tm.that(tm.ok(scoped.resolve(c.Directory.CONFIG)) is scoped.settings, eq=True)
        tm.that(tm.ok(scoped.resolve(c.FIELD_CONTEXT)) is scoped.context, eq=True)
        tm.that(scoped.settings is clean_container.settings, eq=False)

    def test_scope_spec_overrides_inherited_registration(
        self,
        clean_container: p.Container,
    ) -> None:
        """A scope declaration replaces the inherited value of the same name."""
        _ = clean_container.bind("mode", "parent")

        scoped = clean_container.scope(
            registration=m.ServiceRegistrationSpec(services={"mode": "scoped"}),
        )

        tm.ok(scoped.resolve("mode"), eq="scoped")
        tm.ok(clean_container.resolve("mode"), eq="parent")

    def test_auto_registration_without_imported_caller_raises(
        self,
        clean_container: p.Container,
    ) -> None:
        """A caller module absent from ``sys.modules`` is never skipped."""
        caller = types.FunctionType(
            self._shared_with_auto_registration.__code__,
            {"__name__": "tests_unimported_caller", "FlextContainer": FlextContainer},
        )

        with pytest.raises(e.ValidationError, match="caller module"):
            caller()

        tm.that(clean_container.names(), empty=True)

    def test_auto_registration_registers_caller_module_factories(
        self,
        clean_container: p.Container,
    ) -> None:
        """Every ``@d.factory()`` function of the calling module is registered."""
        module = types.ModuleType("tests_auto_registration_caller")

        @d.factory("made")
        def build() -> t.JsonValue:
            return "made-value"

        module.__dict__.update(build=build, FlextContainer=FlextContainer)
        caller = types.FunctionType(
            self._shared_with_auto_registration.__code__,
            module.__dict__,
        )
        sys.modules[module.__name__] = module
        try:
            caller()
        finally:
            del sys.modules[module.__name__]

        tm.ok(clean_container.resolve("made"), eq="made-value")
