"""Registry creation, dispatch, and summary example sections.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples.constants import c
from examples.ex_12_registry_support import (
    ProtocolHandler,
    as_registry_handler,
    discovered_handler,
)
from examples.models import m
from examples.shared import ExamplesFlextShared
from examples.utilities import u
from flext_core import h, r

if TYPE_CHECKING:
    from examples.protocols import p
    from examples.typings import t


class Ex12RegistryFlow(ExamplesFlextShared):
    """Registry flow checks for the executable example."""

    def _exercise_create_and_service_methods(self) -> t.Pair[p.Registry, p.Dispatcher]:
        self.section("create_and_service_methods")
        discovered_value = self.rand_str(4)
        dispatcher = u.build_dispatcher()
        registry = u.build_registry(dispatcher=dispatcher)
        _ = u.build_registry()
        _ = u.build_registry(dispatcher=None)
        _ = u.build_registry(auto_discover_handlers=False)
        _ = u.build_registry(auto_discover_handlers=True)
        self.audit_check(
            "decorated_handler.type",
            type(
                discovered_handler(m.Examples.CommandA(value=discovered_value)),
            ).__name__,
        )
        self.audit_check("execute.success", registry.execute().success)
        return (registry, dispatcher)

    def _exercise_registration_and_dispatch(
        self,
        registry: p.Registry,
        dispatcher: p.Dispatcher,
    ) -> t.Pair[ProtocolHandler, ProtocolHandler]:
        self.section("registration_and_dispatch")
        labels = (self.rand_str(3), self.rand_str(3))
        callable_prefix = self.rand_str(3)
        callable_name = self.rand_str(10)
        command_values = (self.rand_str(6), self.rand_int(1, 100))
        handler_a = ProtocolHandler(labels[0], m.Examples.CommandA)
        handler_b = ProtocolHandler(labels[1], m.Examples.CommandB)
        handler_mode = h.create_from_callable(
            lambda msg: f"{callable_prefix}:{msg!r}",
            handler_name=callable_name,
            handler_type=c.HandlerType.COMMAND,
        )
        self._audit_flow_registrations(
            registry,
            handler_a,
            handler_b,
            handler_mode,
            callable_prefix,
        )
        self._audit_flow_dispatches(
            dispatcher,
            handler_a,
            handler_b,
            labels,
            command_values,
        )
        return (handler_a, handler_b)

    def _audit_flow_registrations(
        self,
        registry: p.Registry,
        handler_a: ProtocolHandler,
        handler_b: ProtocolHandler,
        handler_mode: ProtocolHandler,
        callable_prefix: str,
    ) -> None:
        """Register the flow handlers and audit each registration outcome."""
        reg_one = registry.register_handler(as_registry_handler(handler_a))
        reg_dup = registry.register_handler(as_registry_handler(handler_a))
        reg_two = registry.register_handler(as_registry_handler(handler_b))
        reg_mode = registry.register_handler(as_registry_handler(handler_a))
        self.audit_check("register_handler.a.success", reg_one.success)
        self.audit_check(
            "register_handler.a.id_present",
            bool(reg_one.value.registration_id) if reg_one.success else False,
        )
        self.audit_check("register_handler.duplicate.success", reg_dup.success)
        self.audit_check("register_handler.b.success", reg_two.success)
        self.audit_check("register_handler.mode.success", reg_mode.success)
        handler_mode_probe = handler_mode.handle(callable_prefix)
        self.audit_check(
            "create_from_callable.handle_success",
            handler_mode_probe.success,
        )
        batch = registry.register_handlers([
            as_registry_handler(handler_a),
            as_registry_handler(handler_b),
            as_registry_handler(handler_a),
        ])
        self.audit_check("register_handlers.success", batch.success)
        self.audit_check(
            "register_handlers.registered_len",
            len(batch.value.registered) if batch.success else -1,
        )
        self.audit_check(
            "register_handlers.errors_len",
            len(batch.value.errors) if batch.success else -1,
        )

    def _audit_flow_dispatches(
        self,
        dispatcher: p.Dispatcher,
        handler_a: ProtocolHandler,
        handler_b: ProtocolHandler,
        labels: tuple[str, str],
        command_values: tuple[str, int],
    ) -> None:
        """Dispatch both flow commands and audit the routed values."""
        del handler_a, handler_b
        cmd_a = m.Examples.CommandA(value=command_values[0])
        dispatch_a = dispatcher.dispatch(cmd_a)
        self.audit_check("dispatch.a.success", dispatch_a.success)
        self.audit_check(
            "dispatch.a.value",
            dispatch_a.value == f"{labels[0]}:{command_values[0]}",
        )
        cmd_b = m.Examples.CommandB(amount=command_values[1])
        dispatch_b = dispatcher.dispatch(cmd_b)
        self.audit_check("dispatch.b.success", dispatch_b.success)
        self.audit_check(
            "dispatch.b.value",
            dispatch_b.value == f"{labels[1]}:{command_values[1]}",
        )

    def _exercise_summary_and_mixins(self) -> None:
        self.section("summary_and_mixins")
        self._audit_registry_summary_flags()
        self._audit_result_mixins_and_ensure()
        self._audit_mapping_and_generation()

    def _audit_registry_summary_flags(self) -> None:
        """Audit the RegistrySummary success/failure flag resolution."""
        summary_error = self.rand_str(5)
        summary_ok = m.RegistrySummary()
        summary_fail = m.RegistrySummary(errors=[summary_error])
        ok_success_attr = summary_ok.success
        summary_ok_success = (
            ok_success_attr() if callable(ok_success_attr) else ok_success_attr
        )
        ok_failure_attr = summary_ok.failure
        summary_ok_failure = (
            ok_failure_attr() if callable(ok_failure_attr) else ok_failure_attr
        )
        fail_success_attr = summary_fail.success
        summary_fail_success = (
            fail_success_attr() if callable(fail_success_attr) else fail_success_attr
        )
        fail_failure_attr = summary_fail.failure
        summary_fail_failure = (
            fail_failure_attr() if callable(fail_failure_attr) else fail_failure_attr
        )
        self.audit_check("summary.ok.success", summary_ok_success)
        self.audit_check("summary.ok.failure", summary_ok_failure)
        self.audit_check("summary.fail.success", summary_fail_success)
        self.audit_check("summary.fail.failure", summary_fail_failure)

    def _audit_result_mixins_and_ensure(self) -> None:
        """Audit the result mixins and the ensure-result contract."""
        ok_value = self.rand_str(6)
        fail_message = self.rand_str(7)
        fail_code = self.rand_str(5)
        ensured_raw = self.rand_int(1, 200)
        ensured_existing = self.rand_int(1, 200)
        ok_result = r[str].ok(ok_value)
        fail_result = r[str].fail(fail_message, error_code=fail_code)
        self.audit_check("mixin.ok.unwrap_or", ok_result.value == ok_value)
        self.audit_check("mixin.fail.error", fail_result.error == fail_message)
        self.audit_check("mixin.fail.error_code", fail_result.error_code == fail_code)
        self.audit_check(
            "ensure_result.raw",
            r[int].ok(ensured_raw).value == ensured_raw,
        )
        self.audit_check(
            "ensure_result.existing",
            r[int].ok(ensured_existing).value == ensured_existing,
        )

    def _audit_mapping_and_generation(self) -> None:
        """Audit ConfigMap dumps and the identifier generators."""
        map_key_a = self.rand_str(3)
        map_key_b = self.rand_str(3)
        map_val_a = self.rand_int(1, 9)
        map_val_b = self.rand_str(3)
        handler_name = self.rand_str(6)
        handler_id = self.rand_str(8)
        prefix = f"reg.{self.rand_str(4)}"
        self.audit_check("to_dict.none", m.ConfigMap(root={}))
        self.audit_check(
            "to_dict.mapping",
            m.ConfigMap(root={map_key_a: map_val_a, map_key_b: map_val_b}),
        )
        self.audit_check(
            "to_dict.basemodel",
            m.Handler(handler_name=handler_name, handler_id=handler_id).model_dump(),
        )
        self.audit_check("generate_id.len", len(u.generate_id()))
        self.audit_check(
            "generate_prefixed_id",
            u.generate_prefixed_id(prefix, 6).startswith(f"{prefix}_"),
        )
        self.audit_check(
            "generate_datetime_utc.type",
            type(u.generate_datetime_utc()).__name__,
        )
