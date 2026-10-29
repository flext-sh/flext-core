"""Behavioral tests for the truthful container write rules.

Every registration passes one write path: empty, duplicate and reserved names
raise ``e.ValidationError``; the reserved core services stay private to the
container, including in scopes; factory auto-registration never skips.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flext_core import FlextContainer
from tests import c, d, e, m, p, t, tm, u


class TestsFlextCoreContainerTruthful:
    """Exercise the rules of the single container write path."""

    @staticmethod
    def _shared_with_auto_registration() -> None:
        _ = FlextContainer.shared(auto_register_factories=True)

    @staticmethod
    def _caller_in(module: types.ModuleType) -> types.FunctionType:
        """Rebuild the auto-registration caller so it runs inside ``module``.

        Returns:
            The resulting ``types.FunctionType``.

        """
        module.__dict__["FlextContainer"] = FlextContainer
        return types.FunctionType(
            TestsFlextCoreContainerTruthful._shared_with_auto_registration.__code__,
            module.__dict__,
        )

    @staticmethod
    def _register(
        container: p.Container,
        kind: str,
        name: str,
        value: str,
    ) -> None:
        """Register ``value`` under ``name`` through the public ``kind`` write."""
        factory = u.Tests.create_factory(value)
        if kind == "bind":
            _ = container.bind(name, value)
        elif kind == "factory":
            _ = container.factory(name, factory)
        else:
            _ = container.resource(name, factory)

    @staticmethod
    @pytest.mark.parametrize("name", sorted(c.CONTAINER_RESERVED_NAMES))
    def test_public_writes_reject_reserved_names(
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
        tm.fail(clean_container.has(name))

    @staticmethod
    @pytest.mark.parametrize("second", ["bind", "factory", "resource"])
    @pytest.mark.parametrize("first", ["bind", "factory", "resource"])
    def test_duplicate_is_rejected_across_registration_kinds(
        first: str,
        second: str,
        clean_container: p.Container,
    ) -> None:
        """A registered name cannot be reused by any kind; the first stays."""
        TestsFlextCoreContainerTruthful._register(
            clean_container, first, "shared_name", "first"
        )

        with pytest.raises(e.ValidationError, match="shared_name"):
            TestsFlextCoreContainerTruthful._register(
                clean_container, second, "shared_name", "second"
            )

        tm.ok(clean_container.resolve("shared_name"), eq="first")

    @staticmethod
    def test_resource_empty_name_raises(clean_container: p.Container) -> None:
        """An empty resource name raises instead of being ignored."""
        with pytest.raises(e.ValidationError, match=c.ERR_CONTAINER_NAME_EMPTY):
            _ = clean_container.resource("", u.Tests.create_factory("value"))

    @staticmethod
    def test_drop_of_reserved_name_fails_and_keeps_core_service(
        clean_container: p.Container,
    ) -> None:
        """Core services are not public, so drop reports them as not found."""
        tm.fail(clean_container.drop(c.ServiceName.LOGGER), has="logger")
        tm.ok(clean_container.resolve(c.ServiceName.LOGGER))

    @staticmethod
    def test_scope_keeps_logger_internal(clean_container: p.Container) -> None:
        """Regression: a scope re-registers LOGGER as internal, never public."""
        _ = clean_container.bind("public", "value")

        scoped = clean_container.scope()
        nested = scoped.scope()

        for container in (scoped, nested):
            tm.that(list(container.names()), eq=["public"])
            tm.fail(container.has(c.ServiceName.LOGGER))
            logger: p.RegisterableService = tm.ok(
                container.resolve(c.ServiceName.LOGGER)
            )
            tm.ok(isinstance(logger, p.Logger))
            with pytest.raises(e.ValidationError, match="reserved"):
                _ = container.factory(
                    c.ServiceName.LOGGER,
                    u.Tests.create_factory("value"),
                )

    @staticmethod
    def test_scope_binds_core_services_to_its_own_runtime(
        clean_container: p.Container,
    ) -> None:
        """The scoped settings and context services are the scope's own."""
        scoped = clean_container.scope(subproject="unit")

        tm.ok(tm.ok(scoped.resolve(c.Directory.CONFIG)) is scoped.settings)
        tm.ok(tm.ok(scoped.resolve(c.FIELD_CONTEXT)) is scoped.context)
        tm.fail(scoped.settings is clean_container.settings)

    @staticmethod
    def test_scope_spec_overrides_inherited_registration(
        clean_container: p.Container,
    ) -> None:
        """A scope declaration replaces the inherited value of the same name."""
        _ = clean_container.bind("mode", "parent")

        scoped = clean_container.scope(
            registration=m.ServiceRegistrationSpec(services={"mode": "scoped"}),
        )

        tm.ok(scoped.resolve("mode"), eq="scoped")
        tm.ok(clean_container.resolve("mode"), eq="parent")

    @staticmethod
    def test_mapping_registration_keeps_input_and_serializes_its_payload(
        clean_container: p.Container,
    ) -> None:
        """An explicitly validated model retains the caller's payload values."""
        instant = datetime.now(UTC)
        path = Path("registered-payload")
        payload: t.MutableMetadataMapping = {"created": instant, "path": path}
        _ = clean_container.bind("mapping-payload", m.ConfigMap(root=payload))
        value: p.Model[t.JsonDict] = tm.ok(
            clean_container.resolve("mapping-payload", type_cls=m.ConfigMap)
        )
        tm.that(
            value.model_dump(mode="json"),
            eq=u.normalize_to_json_value(payload),
        )
        tm.ok(payload["created"] is instant)
        tm.ok(payload["path"] is path)

    @staticmethod
    def test_sequence_registration_serializes_a_list_root(
        clean_container: p.Container,
    ) -> None:
        """The registry model port admits list output, not only named fields."""
        instant = datetime.now(UTC)
        path = Path("sequence-payload")
        payload: list[t.JsonPayload] = ["entry", path, instant]
        registered = m.ObjectList(root=payload)
        _ = clean_container.bind("sequence-payload", registered)
        value: p.Model[list[t.JsonValue]] = tm.ok(
            clean_container.resolve("sequence-payload", type_cls=m.ObjectList)
        )

        def resolves_registered(resolved: object) -> bool:
            return resolved is registered

        tm.that(value, where=resolves_registered)
        tm.that(
            value.model_dump(mode="json"),
            eq=u.normalize_to_json_value(payload),
        )
        tm.ok(payload[1] is path)
        tm.ok(payload[2] is instant)

    @staticmethod
    def test_registered_model_member_preserves_its_public_fields(
        clean_container: p.Container,
    ) -> None:
        """A nested Pydantic value retains its own validated serialization."""
        metadata = m.Metadata()
        _ = clean_container.bind(
            "model-payload", m.ConfigMap(root={"metadata": metadata})
        )
        value: p.Model[t.JsonDict] = tm.ok(
            clean_container.resolve("model-payload", type_cls=m.ConfigMap)
        )
        tm.that(
            value.model_dump(mode="json"),
            eq={"metadata": metadata.model_dump(mode="json")},
        )

    @staticmethod
    def test_set_normalization_uses_json_without_an_order_contract() -> None:
        """Set values become JSON arrays without inventing an ordering rule."""
        payload = frozenset({"first", "second"})
        normalized = u.normalize_to_container(payload)

        tm.ok(isinstance(normalized, list))
        if isinstance(normalized, list):
            tm.that(set(normalized), eq=payload)

    @staticmethod
    def test_null_normalization_keeps_the_pydantic_json_value() -> None:
        """An admitted JSON null is data, never an implicit empty string."""
        tm.ok(u.normalize_to_container(None) is None)

    @staticmethod
    def test_unsupported_container_value_is_not_silently_stringified() -> None:
        """Opaque runtime objects fail instead of becoming misleading text."""
        payload = types.ModuleType("unsupported-payload")
        message = c.ERR_RUNTIME_CONTAINER_INVALID_TYPE.format(
            type_name=type(payload).__name__,
        )
        with pytest.raises(TypeError, match=re.escape(message)):
            _ = u.normalize_to_container(payload)

    def test_auto_registration_without_imported_caller_raises(
        self,
        clean_container: p.Container,
    ) -> None:
        """A caller module absent from ``sys.modules`` is never skipped."""
        caller = self._caller_in(types.ModuleType("tests_unimported_caller"))

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

        module.__dict__["build"] = build
        caller = self._caller_in(module)
        sys.modules[module.__name__] = module
        try:
            caller()
        finally:
            del sys.modules[module.__name__]

        tm.ok(clean_container.resolve("made"), eq="made-value")
