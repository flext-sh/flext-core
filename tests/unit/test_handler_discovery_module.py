"""Public decorator/discovery contracts through shared models and real services.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from re import escape
from time import monotonic
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from tests import c, h, m, p, r, tm, u

if TYPE_CHECKING:
    from tests import t


class TestsFlextHandlerDiscovery:
    """One behavioral suite replaces duplicated payload stubs and fixture engines."""

    @staticmethod
    @pytest.mark.parametrize("priority", [0, 1, 10, 100])
    def test_declared_metadata_and_original_callable(priority: int) -> None:
        """Decorator and scanner preserve callable identity and declaration."""
        outcome = r[str].ok("created")

        def operation(_request: p.Routable) -> p.Result[str]:
            return outcome

        decorated = h.handler(command=m.Command, priority=priority)(operation)
        module = u.Tests.handler_module("declared_handlers", operation=decorated)
        name, found, metadata = h.Discovery.scan_module(module)[0]
        tm.that(name, eq="operation")
        tm.that(decorated is operation)
        tm.that(found is operation)
        tm.that(metadata.command is m.Command)
        tm.that(metadata.priority, eq=priority)
        tm.that(metadata.timeout, eq=None)
        tm.that(tuple(metadata.middleware), eq=())
        tm.that(found(m.Command()) is outcome)

    @staticmethod
    @pytest.mark.parametrize("explicit_empty", [False, True])
    def test_defaults_come_from_the_model_owner(*, explicit_empty: bool) -> None:
        """Defaults are derived, never frozen to today's configured values."""

        def operation(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("created")

        decorator = (
            h.handler(command=m.Command, timeout=None, middleware=())
            if explicit_empty
            else h.handler(command=m.Command)
        )
        decorated = decorator(operation)
        metadata = h.Discovery.scan_module(
            u.Tests.handler_module("default_handlers", operation=decorated),
        )[0][2]
        expected = m.DecoratorConfig(command=m.Command)
        tm.that(
            (
                metadata.command,
                metadata.priority,
                metadata.timeout,
                tuple(metadata.middleware),
            ),
            eq=(
                expected.command,
                expected.priority,
                expected.timeout,
                tuple(expected.middleware),
            ),
        )

    @staticmethod
    def test_variadic_declaration_preserves_original_call_signature() -> None:
        """Discovery retains arguments without certifying message dispatchability."""

        def operation(
            request: p.Routable,
            /,
            *suffixes: str,
            prefix: str,
            **labels: str,
        ) -> p.Result[str]:
            return r[str].ok(
                f"{prefix}:{u.resolve_message_route(request)}:{','.join(suffixes)}:"
                f"{labels['label']}"
            )

        decorated = h.handler(command=m.Command)(operation)
        module = ModuleType("variadic_handlers")
        module.operation = decorated
        found = h.Discovery.scan_module(module)[0][1]
        request = m.Command()
        expected = f"start:{u.resolve_message_route(request)}:a,b:end"
        tm.that(decorated is operation)
        tm.that(found is operation)
        tm.that(
            tm.ok(decorated(request, "a", "b", prefix="start", label="end")),
            eq=expected,
        )
        result = found(request, "a", "b", prefix="start", label="end")
        if not isinstance(result, p.ResultView):
            pytest.fail("Discovered handler did not preserve its result contract")
        tm.that(tm.ok(result), eq=expected)

    @staticmethod
    def test_callable_admission_does_not_certify_foreign_return_payload() -> None:
        """The actual dispatcher rejects a foreign callable's unsupported output."""
        module = ModuleType("foreign_handlers")
        received: t.MutableSequenceOf[p.Routable] = []

        def operation(request: p.Routable) -> ModuleType:
            received.append(request)
            return module

        setattr(operation, c.HANDLER_ATTR, m.DecoratorConfig(command=m.Command))
        module.operation = operation
        found = h.Discovery.scan_module(module)[0][1]
        tm.that(found is operation)
        tm.that(received, empty=True)
        dispatcher = u.build_dispatcher()
        report = tm.ok(
            dispatcher.register_handlers((
                m.HandlerRegistration(
                    handler=found, message_type=c.DEFAULT_COMMAND_TYPE
                ),
            ))
        )
        tm.that(report.registered[0].handler is operation)
        request = m.Command()
        outcome = dispatcher.dispatch(request)
        tm.that(outcome.failure)
        tm.that(outcome.error, contains=c.ERR_HANDLER_RETURNED_NON_CONTAINER_VALUE)
        tm.that(received[0] is request)

    @staticmethod
    @pytest.mark.parametrize("timeout", [0.0, 1.0, 30.0])
    def test_unsupported_timeout_fails_before_application(timeout: float) -> None:
        """The executor never accepts a policy it cannot implement."""
        with pytest.raises(ValueError, match="timeout execution is not supported"):
            h.handler(command=m.Command, timeout=timeout)

    @staticmethod
    def test_unsupported_middleware_fails_before_application() -> None:
        """Requested middleware requires an actual executor capability."""
        with pytest.raises(ValueError, match="middleware execution is not supported"):
            h.handler(command=m.Command, middleware=(u.Tests.Middleware,))

    @staticmethod
    def test_negative_priority_fails_at_the_pydantic_boundary() -> None:
        """The model validates the declared nonnegative priority invariant."""

        def operation(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("created")

        with pytest.raises(c.ValidationError):
            h.handler(command=m.Command, priority=-1)(operation)

    @staticmethod
    @pytest.mark.parametrize("command", [m.Command, m.Event])
    def test_stacked_declarations_leave_original_metadata_unchanged(
        command: type,
    ) -> None:
        """Repeated declarations fail without replacing the original metadata."""

        @h.handler(command=m.Command)
        def operation(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("created")

        module = u.Tests.handler_module("single_declaration", operation=operation)
        original = h.Discovery.scan_module(module)[0][2]
        with pytest.raises(ValueError, match="metadata is already declared"):
            h.handler(command=command)(operation)
        tm.that(h.Discovery.scan_module(module)[0][2] is original)

    @staticmethod
    def test_module_filters_private_plain_scalar_and_native_members() -> None:
        """Unrelated builtins without metadata are ignored."""

        @h.handler(command=m.Command)
        def operation(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("created")

        def plain(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("plain")

        module = u.Tests.handler_module(
            "filtered_handlers",
            public=operation,
            _private=operation,
            plain=plain,
            constant=17,
            monotonic=monotonic,
        )
        found = h.Discovery.scan_module(module)
        tm.that(tuple(item[0] for item in found), eq=("public",))
        tm.that(found[0][1] is operation)

    @staticmethod
    def test_module_priority_and_alias_name_order_are_deterministic() -> None:
        """Declared priority then export name determine discovery order."""

        @h.handler(command=m.Command, priority=10)
        def tied(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("tied")

        @h.handler(command=m.Event, priority=100)
        def high(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("high")

        module = u.Tests.handler_module(
            "ordered_handlers", zebra=tied, top=high, alpha=tied
        )
        first = h.Discovery.scan_module(module)
        second = h.Discovery.scan_module(module)
        tm.that(tuple(item[0] for item in first), eq=("top", "alpha", "zebra"))
        tm.that(tuple(item[0] for item in second), eq=tuple(item[0] for item in first))
        tm.that(
            all(left[1] is right[1] for left, right in zip(first, second, strict=True))
        )

    @staticmethod
    @pytest.mark.parametrize(
        "payload", [None, 7, "plain", [1], m.ConfigMap(root={"native": 1})]
    )
    def test_discovery_preserves_raw_payload_identity(payload: t.JsonPayload) -> None:
        """Discovery neither transforms nor executes the original function."""

        @h.handler(command=m.Command)
        def operation(_request: p.Routable) -> t.JsonPayload:
            return payload

        module = u.Tests.handler_module("payload_handlers", operation=operation)
        found = h.Discovery.scan_module(module)[0][1]
        tm.that(found is operation)
        tm.that(found(m.Command()) is payload)

    @staticmethod
    @pytest.mark.parametrize("failure", [False, True])
    def test_result_identity_cause_and_no_late_module_lookup(*, failure: bool) -> None:
        """Saved handlers survive replacement without a module-name lookup."""
        cause = RuntimeError("handler operation failed")
        outcome = (
            r[str].fail("original failure", exception=cause)
            if failure
            else r[str].ok("original")
        )

        @h.handler(command=m.Command)
        def operation(_request: p.Routable) -> p.Result[str]:
            return outcome

        @h.handler(command=m.Command)
        def replacement(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("replacement")

        module = u.Tests.handler_module("unimported_handlers", operation=operation)
        saved = h.Discovery.scan_module(module)[0][1]
        module.operation = replacement
        tm.that(h.Discovery.scan_module(module)[0][1] is replacement)
        del module.operation
        tm.that(h.Discovery.scan_module(module), empty=True)
        tm.that(saved is operation)
        tm.that(saved(m.Command()) is outcome)
        if failure:
            tm.that(outcome.exception is cause)

    @staticmethod
    @pytest.mark.parametrize("invalid", ["invalid", m.ConfigMap(root={})])
    def test_malformed_metadata_fails_before_registration(
        invalid: t.JsonPayload,
    ) -> None:
        """Foreign metadata is rejected rather than treated as absent."""

        def operation(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("created")

        setattr(operation, c.HANDLER_ATTR, invalid)
        with pytest.raises(TypeError):
            h.Discovery.scan_module(
                u.Tests.handler_module("invalid_handlers", operation=operation)
            )

    @staticmethod
    def test_injected_unsupported_policy_is_rejected_by_discovery() -> None:
        """Valid metadata cannot activate unsupported execution policies."""

        def operation(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("created")

        setattr(
            operation, c.HANDLER_ATTR, m.DecoratorConfig(command=m.Command, timeout=1.0)
        )
        with pytest.raises(ValueError, match=escape(c.ERR_HANDLER_POLICY_UNSUPPORTED)):
            h.Discovery.scan_module(
                u.Tests.handler_module("unsupported_handlers", operation=operation)
            )

    @staticmethod
    @pytest.mark.parametrize(
        ("service_type", "names"),
        [
            (u.Tests.HandlerService, ("create", "delete", "query")),
            (u.Tests.InheritedHandlerService, ("create", "delete", "query")),
            (u.Tests.PrivateHandlerService, ("_private",)),
            (u.Tests.DescriptorHandlerService, ("class_method", "static")),
        ],
    )
    def test_class_discovery_uses_real_service_mro_and_descriptors(
        service_type: type,
        names: t.StrSequence,
    ) -> None:
        """Class scanning includes inherited and decorated descriptors only."""
        first = h.Discovery.scan_class(service_type)
        second = h.Discovery.scan_class(service_type)
        tm.that(tuple(name for name, _ in first), eq=names)
        tm.that(tuple(name for name, _ in second), eq=names)
        priorities = [metadata.priority for _, metadata in first]
        tm.that(priorities, eq=sorted(priorities, reverse=True))
        tm.that(h.Discovery.has_handlers(service_type))

    @staticmethod
    def test_class_metadata_and_request_forwarding_are_observable() -> None:
        """Instances execute declared operations and honor MRO overrides."""
        metadata = dict(h.Discovery.scan_class(u.Tests.HandlerService))
        tm.that(metadata["create"].command is m.Command)
        tm.that(metadata["delete"].command is m.Event)
        tm.that(metadata["query"].command is m.Tests.RegistryQuery)
        request = m.Command(command_type="request-forwarded")
        tm.that(
            tm.ok(u.Tests.HandlerService().create(request)), eq=request.command_type
        )
        tm.that(
            tm.ok(u.Tests.InheritedHandlerService().create(request)), eq="overridden"
        )

    @staticmethod
    def test_empty_inputs_do_not_construct_a_service() -> None:
        """Discovery introspects without bootstrap or implicit execution."""
        tm.that(
            h.Discovery.scan_module(u.Tests.handler_module("empty_handlers", value=1)),
            empty=True,
        )
        tm.that(h.Discovery.scan_class(m.Command), empty=True)
        tm.that(not h.Discovery.has_handlers(m.Command))
