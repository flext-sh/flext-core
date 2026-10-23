"""Behavioral tests for typed service operations (ADR-019, ``u.service_operations``).

Asserts the public contract only: which methods ``u.service_operations``
reports as operations, the frozen ``m.ServiceOperation`` values it returns, and
the ``TypeError`` it raises for every malformed shape. Discovery is lazy, so a
malformed service class is created without error and fails only when asked.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, override

import pytest

from tests import c, h, m, p, r, s, t, tm, u

if TYPE_CHECKING:
    from flext_core import FlextSettings


class TestsFlextCoreServiceOperations:
    """Public-contract tests for lazy operation discovery."""

    class OperationService(s[bool]):
        """Instance-owned readiness and response formatting for real operations."""

        ready: bool = True
        response_prefix: str = ""

    class CommandService(OperationService):
        """Service with one request operation and one input-less operation."""

        counter: t.Port[p.Tests.Counter] = m.Field(
            exclude=True,
            description="Counter port, never an operation.",
        )

        @override
        def execute(self) -> p.Result[bool]:
            """Execute is part of the kernel, never an operation.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(value=True)

        @h.handler(command=m.Tests.DispatchRequest)
        def dispatch(self, request: p.Tests.DispatchRequest) -> p.Result[str]:
            """Dispatch one command by name.

            The summary is the first docstring line only.

            Returns:
                The resulting ``p.Result[str]``.

            """
            return r[str].ok(self.response_prefix + request.command_name)

        def status(self) -> p.Result[bool]:
            """Report readiness.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(self.ready)

        @property
        def label(self) -> str:
            """A property, which is not an operation."""
            return self.__class__.__name__

        @classmethod
        def build(cls) -> p.Result[bool]:
            """Provide a classmethod, which is not an operation.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(value=True)

        @staticmethod
        def helper() -> p.Result[bool]:
            """Provide a staticmethod, which is not an operation.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(value=True)

        @m.model_validator(mode="after")
        def check_state(self) -> Self:
            """Provide a Pydantic validator, which is not an operation.

            Returns:
                The resulting ``Self``.

            """
            return self

        @staticmethod
        def _private() -> p.Result[bool]:
            """Provide a private method, which is not an operation.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(value=True)

    class ReadSide(OperationService):
        """Sibling declaring ``run``."""

        def run(self) -> p.Result[bool]:
            """Run the read side.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(self.ready)

    class WriteSide(OperationService):
        """Sibling also declaring ``run``."""

        def run(self) -> p.Result[bool]:
            """Run the write side.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(not self.ready)

    class CollidingService(ReadSide, WriteSide):
        """Composes two siblings that both declare ``run``."""

    class OverridingService(ReadSide):
        """Overrides its parent's operation: not a collision."""

        @override
        def run(self) -> p.Result[bool]:
            """Run the overriding side.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(value=False)

    class EmptyService(s[bool]):
        """Service without operations."""

    class AsyncService(OperationService):
        """Operation declared async."""

        async def fetch(self) -> p.Result[bool]:
            """Fetch asynchronously.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(self.ready)

    class GenericService(OperationService):
        """Operation declaring type parameters."""

        def echo[V: p.Base](self, request: V) -> p.Result[V]:
            """Echo the request generically.

            Returns:
                The resulting ``p.Result[V]``.

            """
            if not self.ready:
                return r[V].fail("Service is not ready")
            return r[V].ok(request)

    class TwoRequestService(OperationService):
        """Operation taking two requests."""

        def join(
            self,
            left: m.Tests.DispatchRequest,
            right: m.Tests.DispatchRequest,
        ) -> p.Result[bool]:
            """Join two requests.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(self.ready and left == right)

    class KeywordOnlyService(OperationService):
        """Operation with a keyword-only request."""

        def send(self, *, request: m.Tests.DispatchRequest) -> p.Result[bool]:
            """Send a request.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(self.ready and bool(request))

    class DefaultRequestService(OperationService):
        """Operation whose request has a default."""

        def send(
            self,
            request: m.Tests.DispatchRequest | None = None,
        ) -> p.Result[bool]:
            """Send an optional request.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(self.ready and request is None)

    class UndocumentedService(OperationService):
        """Operation without a docstring."""

        def _status(self) -> p.Result[bool]:
            return r[bool].ok(self.ready)

        def status(self) -> p.Result[bool]:
            """Expose the operation for the missing-docstring scenario.

            Returns:
                The instance's readiness result.

            """
            return self._status()

        # Discovery must see an undocumented method with its typed binding intact.
        status.__doc__ = None

    class PlainReturnService(OperationService):
        """Operation returning a plain value."""

        def status(self) -> bool:
            """Report readiness.

            Returns:
                The resulting ``bool``.

            """
            return self.ready

    class UnionRequestService(OperationService):
        """Operation whose request annotation is not a dotted name."""

        def send(self, request: m.Tests.DispatchRequest | str) -> p.Result[bool]:
            """Send a request.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(self.ready and bool(request))

    class ScalarRequestService(OperationService):
        """Operation whose request is not a Pydantic model."""

        def lookup(self, request: str) -> p.Result[str]:
            """Look up a name.

            Returns:
                The resulting ``p.Result[str]``.

            """
            return r[str].ok(self.response_prefix + request)

    class TypeCheckingOnlyService(OperationService):
        """Operation whose request is imported only under TYPE_CHECKING."""

        def configure(self, request: FlextSettings) -> p.Result[bool]:
            """Configure from settings.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            return r[bool].ok(self.ready and bool(request))

    class MissingMetadataService(OperationService):
        """A Protocol request without its explicit schema association."""

        def send(self, request: p.Tests.DispatchRequest) -> p.Result[str]:
            """Send the requested command.

            Returns:
                The requested command name in a successful result.

            """
            return r[str].ok(self.response_prefix + request.command_name)

    class AliasedRequestService(OperationService):
        """A quoted t alias resolves to the same Protocol without eval."""

        @h.handler(command=m.Tests.DispatchRequest)
        def send(self, request: t.Core.Tests.DispatchRequest) -> p.Result[str]:
            """Send the requested command.

            Returns:
                The requested command name in a successful result.

            """
            return r[str].ok(self.response_prefix + request.command_name)

    class ConcreteRequestService(OperationService):
        """A model annotation cannot replace the public Protocol contract."""

        @h.handler(command=m.Tests.DispatchRequest)
        def send(self, request: m.Tests.DispatchRequest) -> p.Result[str]:
            """Send the requested command.

            Returns:
                The requested command name in a successful result.

            """
            return r[str].ok(self.response_prefix + request.command_name)

    class MismatchedMetadataService(OperationService):
        """The explicit schema must provide every consumed Protocol member."""

        @h.handler(command=m.Value)
        def send(self, request: p.Tests.DispatchRequest) -> p.Result[str]:
            """Send the requested command.

            Returns:
                The requested command name in a successful result.

            """
            # mismatched metadata path
            return r[str].ok(self.response_prefix + request.command_name)

    class NonModelMetadataService(OperationService):
        """A metadata command class is not necessarily a request model."""

        @h.handler(command=str)
        def send(self, request: p.Tests.DispatchRequest) -> p.Result[str]:
            """Send the requested command.

            Returns:
                The requested command name in a successful result.

            """
            return r[str].ok(self.response_prefix + request.command_name)

    class InputlessMetadataService(OperationService):
        """An input-less operation cannot consume declared command metadata."""

        @h.handler(command=m.Tests.DispatchRequest)
        def send(self) -> p.Result[str]:
            """Report readiness.

            Returns:
                A successful readiness result.

            """
            return r[str].ok(self.response_prefix + "ready")

    # --- Discovery -------------------------------------------------------

    def test_discovers_request_and_input_less_operations(self) -> None:
        """Only the two public operations are reported, sorted by name."""
        operations = u.service_operations(self.CommandService)

        tm.that(
            [(op.name, op.summary, op.request) for op in operations],
            eq=[
                ("dispatch", "Dispatch one command by name.", m.Tests.DispatchRequest),
                ("status", "Report readiness.", None),
            ],
        )
        tm.ok(operations[0].handler is self.CommandService.dispatch)

    @staticmethod
    @pytest.mark.parametrize(
        ("service_type", "defect"),
        [
            (MissingMetadataService, "requires typed h.handler"),
            (ConcreteRequestService, "Protocol"),
            (MismatchedMetadataService, "command_name"),
            (NonModelMetadataService, "Pydantic request model"),
            (InputlessMetadataService, "input-less operation"),
        ],
    )
    def test_request_metadata_has_one_explicit_contract(
        service_type: type[p.Service[bool]],
        defect: str,
    ) -> None:
        """Invalid schema associations fail lazily without annotation inference."""
        with pytest.raises(TypeError, match=defect):
            u.service_operations(service_type)

    def test_operations_are_frozen_models(self) -> None:
        """A reported ``m.ServiceOperation`` rejects assignment: it is frozen."""
        operation = u.service_operations(self.CommandService)[0]
        field_name = next(iter(m.ServiceOperation.model_fields))

        with pytest.raises(m.ValidationError):
            setattr(operation, field_name, operation.summary)

    def test_quoted_protocol_alias_uses_explicit_model_metadata(self) -> None:
        """Quoted aliases do not turn the annotation into a model lookup route."""
        (operation,) = u.service_operations(self.AliasedRequestService)
        tm.ok(operation.request is m.Tests.DispatchRequest)
        tm.ok(operation.handler is self.AliasedRequestService.send)

    def test_discovery_is_cached_per_class(self) -> None:
        """A second call returns the same operations for the same class."""
        first = u.service_operations(self.CommandService)

        tm.ok(u.service_operations(self.CommandService) is first)

    def test_override_of_a_parent_operation_is_not_a_collision(self) -> None:
        """A subclass overriding its parent's operation keeps one operation."""
        operations = u.service_operations(self.OverridingService)

        tm.that(
            [(op.name, op.summary) for op in operations],
            eq=[("run", "Run the overriding side.")],
        )

    def test_malformed_service_class_is_created_without_error(self) -> None:
        """Discovery is lazy: class creation never checks operation shape."""
        tm.ok(self.UndocumentedService().status().unwrap())

    # --- Failures --------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        ("service_type", "operation", "defect"),
        [
            (AsyncService, "fetch", c.ERR_SERVICE_OPERATION_ASYNC),
            (GenericService, "echo", c.ERR_SERVICE_OPERATION_GENERIC),
            (TwoRequestService, "join", c.ERR_SERVICE_OPERATION_SIGNATURE),
            (KeywordOnlyService, "send", c.ERR_SERVICE_OPERATION_SIGNATURE),
            (DefaultRequestService, "send", c.ERR_SERVICE_OPERATION_SIGNATURE),
            (UndocumentedService, "status", c.ERR_SERVICE_OPERATION_DOCSTRING),
            (PlainReturnService, "status", c.ERR_SERVICE_OPERATION_RESULT),
            (UnionRequestService, "send", c.ERR_SERVICE_OPERATION_NAME),
            (ScalarRequestService, "lookup", c.ERR_SERVICE_OPERATION_REQUEST),
            (TypeCheckingOnlyService, "configure", c.ERR_SERVICE_OPERATION_UNBOUND),
        ],
    )
    def test_malformed_operation_raises_with_operation_module_and_fix(
        service_type: type[s[bool]],
        operation: str,
        defect: str,
    ) -> None:
        """Each malformed shape names the operation, module and fix."""
        with pytest.raises(TypeError) as raised:
            u.service_operations(service_type)

        message = str(raised.value)
        tm.that(message, has=f"{service_type.__qualname__}.{operation} ")
        tm.that(message, has=f"(module {__name__}, annotation ")
        tm.ok(message.endswith(defect))

    def test_type_checking_only_import_names_the_annotation(self) -> None:
        """An unbound request annotation is reported verbatim."""
        with pytest.raises(TypeError) as raised:
            u.service_operations(self.TypeCheckingOnlyService)

        tm.that(
            str(raised.value),
            eq=c.ERR_SERVICE_OPERATION.format(
                service=self.TypeCheckingOnlyService.__qualname__,
                operation="configure",
                module=__name__,
                annotation="FlextSettings",
                defect=c.ERR_SERVICE_OPERATION_UNBOUND,
            ),
        )

    def test_sibling_name_collision_raises(self) -> None:
        """Two sibling classes declaring one operation name fail discovery."""
        with pytest.raises(TypeError) as raised:
            u.service_operations(self.CollidingService)

        owners = f"{self.ReadSide.__qualname__}, {self.WriteSide.__qualname__}"
        tm.that(
            str(raised.value).endswith(
                c.ERR_SERVICE_OPERATION_COLLISION.format(owners=owners),
            ),
            eq=True,
        )

    def test_service_without_operations_raises(self) -> None:
        """A service whose only public method is the kernel's has no operation."""
        with pytest.raises(TypeError) as raised:
            u.service_operations(self.EmptyService)

        tm.that(
            str(raised.value),
            eq=c.ERR_SERVICE_NO_OPERATIONS.format(
                service=self.EmptyService.__qualname__,
                module=__name__,
            ),
        )
