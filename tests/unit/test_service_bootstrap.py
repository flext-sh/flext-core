"""Behavioral tests for the service bootstrap and port contract (ADR-019).

Asserts the observable public contract only: a service's ``execute`` result, the
options ``u.resolve_runtime_options`` returns for every supported source, the
runtime a service builds, and the validation of ports and runtime seeds. No
private attribute access, no collaborator spying, no internal patching.
"""

from __future__ import annotations

from typing import override

import pytest
from flext_tests import r, tm

from flext_core import FlextContext, FlextSettings
from tests.base import s
from tests.constants import c
from tests.models import m
from tests.protocols import p
from tests.typings import t
from tests.utilities import u


class TestsFlextCoreServiceBootstrap:
    """Public-contract tests for service execution, runtime options and ports."""

    class ConcreteTestService(s[bool]):
        """Concrete service whose execute contract yields a successful result."""

        @override
        def execute(self) -> p.Result[bool]:
            return r[bool].ok(True)

    # --- Service execution contract ------------------------------------

    def test_execute_returns_successful_result_with_payload(self) -> None:
        """A concrete service returns its successful public payload."""
        result = self.ConcreteTestService().execute()

        tm.that(result.success, eq=True)
        tm.that(result.value, eq=True)

    def test_execute_result_unwraps_to_payload(self) -> None:
        """The service result unwraps to its public payload."""
        result = self.ConcreteTestService().execute()

        tm.that(result.unwrap(), eq=True)

    # --- resolve_runtime_options ---------------------------------------

    def test_resolve_with_no_source_yields_empty_options(self) -> None:
        """Missing bootstrap input resolves to options that inject nothing."""
        resolved = u.resolve_runtime_options()

        tm.that(dict(resolved), eq=dict(m.RuntimeBootstrapOptions()))

    def test_resolve_returns_supplied_model_unchanged(self) -> None:
        """A supplied options model is the resolved options."""
        options = m.RuntimeBootstrapOptions(settings_overrides={"app_name": "keep"})

        resolved = u.resolve_runtime_options(options)

        tm.that(resolved is options, eq=True)

    def test_runtime_options_accepts_settings_class_contract(self) -> None:
        """Settings class validation uses its method-only class protocol."""
        options = m.RuntimeBootstrapOptions(settings_type=FlextSettings)

        tm.that(options.settings_type is FlextSettings, eq=True)

    def test_service_base_hook_decides_the_settings_type(self) -> None:
        """The hook of the service base supplies the runtime settings class."""
        service = self.ConcreteTestService()
        declared = self.ConcreteTestService.runtime_bootstrap_options().settings_type

        resolved = u.resolve_runtime_options(service)

        tm.that(declared is not None, eq=True)
        tm.that(resolved.settings_type is declared, eq=True)
        tm.that(isinstance(service.settings, FlextSettings), eq=True)
        tm.that(type(service.settings) is declared, eq=True)

    def test_instance_seed_wins_over_the_hook(self) -> None:
        """A settings class seeded on the instance overrides the hook."""
        service = self.ConcreteTestService(settings_type=FlextSettings)

        resolved = u.resolve_runtime_options(service)

        tm.that(resolved.settings_type is FlextSettings, eq=True)
        tm.that(type(service.settings) is FlextSettings, eq=True)

    def test_context_seed_becomes_the_runtime_context(self) -> None:
        """An initial context seeded on the instance is the runtime context."""
        context = FlextContext.create()

        service = self.ConcreteTestService(initial_context=context)

        tm.that(service.context is context, eq=True)

    def test_settings_seed_rejects_a_non_settings_value(self) -> None:
        """A runtime settings seed that is not ``p.Settings`` fails validation."""
        with pytest.raises(m.ValidationError):
            self.ConcreteTestService.model_validate({
                "runtime_settings": c.Tests.DEFAULT_ERROR_MESSAGE
            })

    def test_build_service_runtime_binds_the_container_collaborators(self) -> None:
        """The runtime uses the container's context and command bus."""
        declared = self.ConcreteTestService.runtime_bootstrap_options().settings_type

        runtime = u.build_service_runtime(self.ConcreteTestService())

        tm.that(type(runtime.settings) is declared, eq=True)
        tm.that(runtime.context is runtime.container.context, eq=True)
        tm.that(runtime.dispatcher is runtime.container.dispatcher().unwrap(), eq=True)

    # --- Ports -----------------------------------------------------------

    def test_port_accepts_a_conforming_adapter(self) -> None:
        """A service executes through a real adapter of its port."""
        service = u.Tests.CountingService(counter=u.Tests.MemoryCounter())

        tm.that(service.execute().unwrap(), eq=1)
        tm.that(service.execute().unwrap(), eq=2)

    def test_port_rejects_a_non_conforming_value_on_construction(self) -> None:
        """A value that does not satisfy the port protocol fails validation."""
        with pytest.raises(m.ValidationError):
            u.Tests.CountingService.model_validate({
                "counter": c.Tests.DEFAULT_ERROR_MESSAGE
            })

    def test_port_rejects_a_non_conforming_value_on_assignment(self) -> None:
        """Assigning a non-conforming value to a port fails validation."""
        service = u.Tests.CountingService(counter=u.Tests.MemoryCounter())
        port_name = next(
            iter(
                u.Tests.CountingService.model_fields.keys()
                - self.ConcreteTestService.model_fields.keys()
            )
        )

        with pytest.raises(m.ValidationError):
            setattr(service, port_name, c.Tests.DEFAULT_ERROR_MESSAGE)

    def test_fetch_global_of_a_port_service_raises(self) -> None:
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

    def test_subscripted_port_type_is_rejected_at_class_creation(self) -> None:
        """A port typed by a subscripted generic cannot be validated."""
        with pytest.raises(TypeError) as raised:
            u.create_model(
                "SubscriptedPortService",
                __base__=self.ConcreteTestService,
                result=(
                    t.Port[p.Result[int]],
                    m.Field(exclude=True, description="Subscripted generic port."),
                ),
            )

        tm.that(
            str(raised.value),
            eq=c.ERR_SERVICE_PORT_TYPE.format(
                service="SubscriptedPortService",
                field="result",
                port_type=p.Result[int],
            ),
        )

    def test_concrete_port_type_is_rejected_at_class_creation(self) -> None:
        """A port typed by a concrete class is not a port."""
        with pytest.raises(TypeError) as raised:
            u.create_model(
                "ConcretePortService",
                __base__=self.ConcreteTestService,
                counter=(
                    t.Port[u.Tests.MemoryCounter],
                    m.Field(exclude=True, description="Concrete class port."),
                ),
            )

        tm.that(
            str(raised.value),
            eq=c.ERR_SERVICE_PORT_TYPE.format(
                service="ConcretePortService",
                field="counter",
                port_type=u.Tests.MemoryCounter,
            ),
        )
