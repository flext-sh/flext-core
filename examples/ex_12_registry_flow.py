"""Explicit discovery, real dispatch, and atomic binding examples.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from examples import (
    ExamplesFlextShared,
    c,
    ex_12_registry_support as support,
    h,
    m,
    p,
    r,
    t,
    u,
)


class ExamplesFlextRegistryFlow(ExamplesFlextShared):
    """Exercise the public registry and dispatcher as one explicit graph."""

    def _exercise_discovery(self) -> p.Registry:
        self.section("explicit_discovery")
        dispatcher = u.build_dispatcher()
        registry = u.build_registry(dispatcher, discovery_module=support)
        self.audit_check("dispatcher.identity", registry.dispatcher is dispatcher)
        self.audit_check("execute.success", registry.execute().value)
        discovered = h.Discovery.scan_module(support)
        self.audit_check(
            "discovery.callable_identity",
            any(handler is support.discovered_handler for _, handler, _ in discovered),
        )
        command = m.Examples.CommandA(value=self.rand_str(6))
        self.audit_check(
            "dispatch.model_identity", dispatcher.dispatch(command).value is command
        )
        amount = self.rand_int(1, 100)
        self.audit_check(
            "dispatch.result_payload",
            dispatcher.dispatch(m.Examples.CommandB(amount=amount)).value == amount,
        )
        repeated = registry.register_handlers([
            handler for _, handler, _ in discovered
        ]).value
        self.audit_check("rediscovery.no_new_entries", len(repeated.registered) == 0)
        self.audit_check(
            "rediscovery.exact_skipped",
            tuple(repeated.skipped)
            == tuple(
                u.resolve_message_route(metadata.command)
                for _, _, metadata in discovered
            ),
        )
        self.audit_check(
            "rediscovery.model_identity", dispatcher.dispatch(command).value is command
        )
        callable_handler = h.create_from_callable(
            lambda value: value,
            handler_name=self.rand_str(8),
            handler_type=c.HandlerType.COMMAND,
        )
        value = self.rand_str(6)
        self.audit_check(
            "create_from_callable.payload",
            callable_handler.handle(value).value == value,
        )
        return registry

    def _exercise_atomic_bindings(self) -> None:
        self.section("atomic_bindings")
        registry = u.build_registry()
        command = m.Examples.CommandA(value=self.rand_str(6))
        amount = m.Examples.CommandB(amount=self.rand_int(1, 100))
        first = registry.register_handler(support.discovered_handler)
        self.audit_check("register_handler.success", first.success)
        self.audit_check(
            "register_handler.id_present", bool(first.value.registration_id)
        )
        repeated = registry.register_bindings({
            m.Examples.CommandA: support.discovered_handler
        }).value
        self.audit_check(
            "binding.duplicate_exact",
            len(repeated.registered) == 0
            and tuple(repeated.skipped) == (u.resolve_message_route(command),),
        )
        contradictory = registry.register_bindings({
            m.Examples.CommandB: support.discovered_handler
        })
        self.audit_check("binding.metadata_conflict", contradictory.failure)

        def conflicting_handler(_message: p.Routable) -> t.JsonPayload:
            """Represent a distinct callable competing for an occupied route."""
            return "conflicting handler"

        failed = registry.register_bindings({
            m.Examples.CommandB: support.amount_handler,
            m.Examples.CommandA: conflicting_handler,
        })
        self.audit_check("batch.conflict", failed.failure)
        self.audit_check(
            "batch.no_partial_route", registry.dispatcher.dispatch(amount).failure
        )
        self.audit_check(
            "batch.original_preserved",
            registry.dispatcher.dispatch(command).value is command,
        )
        added = registry.register_bindings({
            m.Examples.CommandB: support.amount_handler
        }).value
        self.audit_check(
            "binding.exact_registered",
            tuple(entry.registration_id for entry in added.registered)
            == (u.resolve_message_route(amount),)
            and len(added.skipped) == 0,
        )
        self.audit_check(
            "binding.dispatch",
            registry.dispatcher.dispatch(amount).value == amount.amount,
        )
        cause = RuntimeError("Handler rejected the command")

        def failed_handler(_message: p.Routable) -> p.Result[t.JsonPayload]:
            """Preserve the original domain failure through dispatch."""
            return r[t.JsonPayload].fail(str(cause), exception=cause)

        failed_registry = u.build_registry()
        failed_registry.register_bindings({
            m.Examples.CommandB: failed_handler
        }).unwrap()
        outcome = failed_registry.dispatcher.dispatch(amount)
        self.audit_check("dispatch.failure", outcome.failure)
        self.audit_check("dispatch.cause", outcome.exception is cause)
