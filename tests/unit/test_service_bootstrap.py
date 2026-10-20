"""Behavioral tests for the service bootstrap and port contract (ADR-019).

Asserts the observable public contract only: a service's ``execute`` result, the
options ``u.resolve_runtime_options`` returns for every supported source, the
runtime a service builds, and the validation of ports and runtime seeds. No
private attribute access, no collaborator spying, no internal patching.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

import pytest

from flext_core import FlextContainer, FlextContext, FlextSettings
from tests import c, m, p, r, s, t, tm, u


class TestsFlextCoreServiceBootstrap:
    """Public-contract tests for service execution, runtime options and ports."""

    class ConcreteTestService(s[bool]):
        """Concrete service whose execute contract yields a successful result."""

        @override
        def execute(self) -> p.Result[bool]:
            return r[bool].ok(value=True)

    class ContextConfiguredService(ConcreteTestService):
        """Service electing its collaborators through the typed bootstrap hook."""

        @classmethod
        @override
        def runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
            source = FlextContainer.shared().scope()
            source.bind("context.selected", "declared-container")
            return m.RuntimeBootstrapOptions(
                context=FlextContext.create(source="declared-context"),
                container=source,
            )

    # --- Service execution contract ------------------------------------

    def test_execute_returns_successful_result_with_payload(self) -> None:
        """A concrete service returns its successful public payload."""
        result = self.ConcreteTestService().execute()

        tm.ok(result.success)
        tm.ok(result.value)

    def test_execute_result_unwraps_to_payload(self) -> None:
        """The service result unwraps to its public payload."""
        result = self.ConcreteTestService().execute()

        tm.ok(result.unwrap())

    # --- resolve_runtime_options ---------------------------------------

    @staticmethod
    def test_resolve_with_no_source_yields_empty_options() -> None:
        """Missing bootstrap input resolves to options that inject nothing."""
        resolved = u.resolve_runtime_options()
        declared = m.RuntimeBootstrapOptions()
        tm.ok(resolved.settings is declared.settings)
        tm.ok(resolved.settings_type is declared.settings_type)
        tm.that(resolved.settings_overrides, eq=declared.settings_overrides)
        tm.ok(resolved.context is declared.context)
        tm.ok(resolved.container is declared.container)
        tm.ok(resolved.dispatcher is declared.dispatcher)

    @staticmethod
    def test_resolve_returns_supplied_model_unchanged() -> None:
        """A supplied options model is the resolved options."""
        options = m.RuntimeBootstrapOptions(settings_overrides={"app_name": "keep"})

        resolved = u.resolve_runtime_options(options)

        tm.ok(resolved is options)

    @staticmethod
    def test_runtime_options_accepts_settings_class_contract() -> None:
        """Settings class validation uses its method-only class protocol."""
        options = m.RuntimeBootstrapOptions(settings_type=FlextSettings)

        tm.ok(options.settings_type is FlextSettings)

    def test_service_base_hook_decides_the_settings_type(self) -> None:
        """The hook of the service base supplies the runtime settings class."""
        service = self.ConcreteTestService()
        declared = self.ConcreteTestService.runtime_bootstrap_options().settings_type

        resolved = u.resolve_runtime_options(service)

        declared = tm.not_none(declared)
        tm.ok(resolved.settings_type is declared)
        tm.ok(service.settings is declared.fetch_global())

    def test_instance_seed_wins_over_the_hook(self) -> None:
        """A settings class seeded on the instance overrides the hook."""
        service = self.ConcreteTestService(settings_type=FlextSettings)

        resolved = u.resolve_runtime_options(service)

        tm.ok(resolved.settings_type is FlextSettings)
        tm.ok(type(service.settings) is FlextSettings)

    def test_context_seed_becomes_the_runtime_context(self) -> None:
        """An initial context seeded on the instance is the runtime context."""
        context = FlextContext.create()

        service = self.ConcreteTestService(initial_context=context)

        tm.ok(service.context is context)

    def test_typed_bootstrap_hook_elects_context_and_container(self) -> None:
        """Custom context/container selection uses public options, not private hooks."""
        service = self.ContextConfiguredService()
        tm.ok(service.context.get("source"), eq="declared-context")
        tm.ok(service.container.resolve("context.selected"), eq="declared-container")
        tm.ok(service.container.context is service.context)
        seeded = FlextContext.create(source="instance-context")
        explicit = self.ContextConfiguredService(initial_context=seeded)
        tm.ok(explicit.context is seeded)
        tm.ok(explicit.container.resolve("context.selected"), eq="declared-container")

    def test_settings_seed_rejects_a_non_settings_value(self) -> None:
        """A runtime settings seed that is not ``p.Settings`` fails validation."""
        with pytest.raises(m.ValidationError):
            self.ConcreteTestService.model_validate({
                "runtime_settings": c.Tests.DEFAULT_ERROR_MESSAGE,
            })

    def test_build_service_runtime_binds_the_container_collaborators(self) -> None:
        """The runtime uses the container's context and command bus."""
        declared = self.ConcreteTestService.runtime_bootstrap_options().settings_type

        runtime = u.build_service_runtime(self.ConcreteTestService())

        declared = tm.not_none(declared)
        tm.ok(runtime.settings is declared.fetch_global())
        tm.ok(runtime.context is runtime.container.context)
        tm.ok(runtime.settings is runtime.container.settings)
        tm.ok(runtime.dispatcher is runtime.container.dispatcher().unwrap())

    @staticmethod
    def test_logging_rebind_preserves_public_di_and_actual_provider_identity() -> None:
        """A logging election scopes the supplied graph, not a fresh shared graph."""
        original_settings, original_options = u.logging_settings(), u.logging_options()
        source = FlextContainer.shared().scope()
        selected_sink = m.ConfigMap(root={"destination": "selected"})
        source.bind("logging.selected_sink", selected_sink)
        context = FlextContext.create()
        dispatcher = source.dispatcher().unwrap()
        provider = FlextSettings.fetch_global().clone(log_level=c.LogLevel.ERROR)
        initial = u.build_service_runtime(
            m.RuntimeBootstrapOptions(
                settings=provider,
                container=source,
                context=context,
                dispatcher=dispatcher,
            )
        )
        if not isinstance(initial.settings, p.LoggingSettings):
            pytest.fail("The actual scoped settings must implement the logging port")
        resolved = u.resolve_logging_settings(
            runtime_settings=initial.settings, debug=True
        )
        rebound = u.build_service_runtime(
            m.RuntimeBootstrapOptions(
                settings=resolved,
                container=initial.container,
                context=initial.context,
                dispatcher=initial.dispatcher,
            )
        )
        if not isinstance(rebound.settings, p.LoggingSettings):
            pytest.fail("The rebound provider must implement the logging port")
        try:
            u.configure_structlog(runtime_settings=rebound.settings)
            tm.ok(u.logging_settings() is rebound.settings)
            tm.ok(rebound.settings is rebound.container.settings)
            tm.that(
                rebound.container.resolve(c.Directory.CONFIG).unwrap()
                is rebound.settings,
                eq=True,
            )
            tm.that(
                rebound.container.resolve("logging.selected_sink").unwrap()
                is selected_sink,
                eq=True,
            )
            tm.ok(rebound.context is context)
            tm.ok(rebound.dispatcher is dispatcher)
            tm.ok(rebound.container.dispatcher().unwrap() is dispatcher)
            tm.that(initial.settings.log_level, eq=provider.log_level)
            tm.that(rebound.settings.log_level, eq=c.LogLevel.DEBUG)
        finally:
            u.configure_structlog(
                settings=original_options, runtime_settings=original_settings
            )

    # --- Ports -----------------------------------------------------------

    @staticmethod
    def test_port_accepts_a_conforming_adapter() -> None:
        """A service executes through a real adapter of its port."""
        service = u.Tests.CountingService(counter=u.Tests.MemoryCounter())

        tm.that(service.execute().unwrap(), eq=1)
        tm.that(service.execute().unwrap(), eq=2)

    @staticmethod
    def test_port_rejects_a_non_conforming_value_on_construction() -> None:
        """A value that does not satisfy the port protocol fails validation."""
        with pytest.raises(m.ValidationError):
            u.Tests.CountingService.model_validate({
                "counter": c.Tests.DEFAULT_ERROR_MESSAGE,
            })

    def test_port_rejects_a_non_conforming_value_on_assignment(self) -> None:
        """Assigning a non-conforming value to a port fails validation."""
        service = u.Tests.CountingService(counter=u.Tests.MemoryCounter())
        port_name = next(
            iter(
                u.Tests.CountingService.model_fields.keys()
                - self.ConcreteTestService.model_fields.keys(),
            ),
        )

        with pytest.raises(m.ValidationError):
            setattr(service, port_name, c.Tests.DEFAULT_ERROR_MESSAGE)

    @staticmethod
    def test_fetch_global_of_a_port_service_raises() -> None:
        """A service with a required port has no argument-free singleton."""
        with pytest.raises(m.ValidationError):
            u.Tests.CountingService.fetch_global()

    def test_service_schema_lists_only_data_fields(self) -> None:
        """Ports and runtime seeds never enter the service JSON Schema."""
        port_schema = u.Tests.CountingService.model_json_schema()
        data_schema = u.Tests.ValidatingService.model_json_schema()

        tm.that(port_schema.get("properties", {}), eq={})
        tm.that(
            set(data_schema["properties"]),
            eq=set(u.Tests.ValidatingService.model_fields)
            - set(self.ConcreteTestService.model_fields),
        )

    @staticmethod
    def test_subscripted_port_type_is_rejected_at_class_creation() -> None:
        """A port typed by a subscripted generic cannot be validated."""
        try:

            class SubscriptedPortService(
                TestsFlextCoreServiceBootstrap.ConcreteTestService,
            ):
                result: t.Port[p.Result[int]] = m.Field(
                    exclude=True,
                    description="Subscripted generic port.",
                )

        except TypeError as raised:
            tm.that(
                str(raised),
                eq=c.ERR_SERVICE_PORT_TYPE.format(
                    service="SubscriptedPortService",
                    field="result",
                    port_type=p.Result[int],
                ),
            )
        else:
            pytest.fail(
                f"{SubscriptedPortService.__name__} was unexpectedly constructed",
            )

    @staticmethod
    def test_concrete_port_type_is_rejected_at_class_creation() -> None:
        """A port typed by a concrete class is not a port."""
        try:

            class ConcretePortService(
                TestsFlextCoreServiceBootstrap.ConcreteTestService,
            ):
                counter: t.Port[u.Tests.MemoryCounter] = m.Field(
                    exclude=True,
                    description="Concrete class port.",
                )

        except TypeError as raised:
            tm.that(
                str(raised),
                eq=c.ERR_SERVICE_PORT_TYPE.format(
                    service="ConcretePortService",
                    field="counter",
                    port_type=u.Tests.MemoryCounter,
                ),
            )
        else:
            pytest.fail(f"{ConcretePortService.__name__} was unexpectedly constructed")
