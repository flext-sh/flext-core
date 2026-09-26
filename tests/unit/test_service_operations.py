"""Behavioral tests for typed service operations (ADR-019, ``u.service_operations``).

Asserts the public contract only: which methods ``u.service_operations``
reports as operations, the frozen ``m.ServiceOperation`` values it returns, and
the ``TypeError`` it raises for every malformed shape. Discovery is lazy, so a
malformed service class is created without error and fails only when asked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, override

import pytest
from flext_tests import r, tm

from tests.base import s
from tests.constants import c
from tests.models import m
from tests.protocols import p
from tests.typings import t
from tests.utilities import u

from ._service_operations_support import TestsFlextCoreServiceOperationsEvaluated

if TYPE_CHECKING:
    from flext_core import FlextSettings


class TestsFlextCoreServiceOperations:
    """Public-contract tests for lazy operation discovery."""

    class CommandService(s[bool]):
        """Service with one request operation and one input-less operation."""

        counter: t.Port[p.Tests.Counter] = m.Field(
            exclude=True, description="Counter port, never an operation."
        )

        @override
        def execute(self) -> p.Result[bool]:
            """Execute is part of the kernel, never an operation."""
            return r[bool].ok(True)

        def dispatch(self, request: m.Tests.DispatchRequest) -> p.Result[str]:
            """Dispatch one command by name.

            The summary is the first docstring line only.
            """
            return r[str].ok(request.command_name)

        def status(self) -> p.Result[bool]:
            """Report readiness."""
            return r[bool].ok(True)

        @property
        def label(self) -> str:
            """A property is not an operation."""
            return self.__class__.__name__

        @classmethod
        def build(cls) -> p.Result[bool]:
            """A classmethod is not an operation."""
            return r[bool].ok(True)

        @staticmethod
        def helper() -> p.Result[bool]:
            """A staticmethod is not an operation."""
            return r[bool].ok(True)

        @m.model_validator(mode="after")
        def check_state(self) -> Self:
            """A Pydantic validator is not an operation."""
            return self

        def _private(self) -> p.Result[bool]:
            """A private method is not an operation."""
            return r[bool].ok(True)

    class ReadSide(s[bool]):
        """Sibling declaring ``run``."""

        def run(self) -> p.Result[bool]:
            """Run the read side."""
            return r[bool].ok(True)

    class WriteSide(s[bool]):
        """Sibling also declaring ``run``."""

        def run(self) -> p.Result[bool]:
            """Run the write side."""
            return r[bool].ok(False)

    class CollidingService(ReadSide, WriteSide):
        """Composes two siblings that both declare ``run``."""

    class OverridingService(ReadSide):
        """Overrides its parent's operation: not a collision."""

        @override
        def run(self) -> p.Result[bool]:
            """Run the overriding side."""
            return r[bool].ok(False)

    class EmptyService(s[bool]):
        """Service without operations."""

    class AsyncService(s[bool]):
        """Operation declared async."""

        async def fetch(self) -> p.Result[bool]:
            """Fetch asynchronously."""
            return r[bool].ok(True)

    class GenericService(s[bool]):
        """Operation declaring type parameters."""

        def echo[V: p.Base](self, request: V) -> p.Result[V]:
            """Echo the request generically."""
            return r[V].ok(request)

    class TwoRequestService(s[bool]):
        """Operation taking two requests."""

        def join(
            self, left: m.Tests.DispatchRequest, right: m.Tests.DispatchRequest
        ) -> p.Result[bool]:
            """Join two requests."""
            return r[bool].ok(left == right)

    class KeywordOnlyService(s[bool]):
        """Operation with a keyword-only request."""

        def send(self, *, request: m.Tests.DispatchRequest) -> p.Result[bool]:
            """Send a request."""
            return r[bool].ok(bool(request))

    class DefaultRequestService(s[bool]):
        """Operation whose request has a default."""

        def send(
            self, request: m.Tests.DispatchRequest | None = None
        ) -> p.Result[bool]:
            """Send an optional request."""
            return r[bool].ok(request is None)

    class UndocumentedService(s[bool]):
        """Operation without a docstring."""

        def status(self) -> p.Result[bool]:
            return r[bool].ok(True)

    class PlainReturnService(s[bool]):
        """Operation returning a plain value."""

        def status(self) -> bool:
            """Report readiness."""
            return True

    class UnionRequestService(s[bool]):
        """Operation whose request annotation is not a dotted name."""

        def send(self, request: m.Tests.DispatchRequest | str) -> p.Result[bool]:
            """Send a request."""
            return r[bool].ok(bool(request))

    class ScalarRequestService(s[bool]):
        """Operation whose request is not a Pydantic model."""

        def lookup(self, request: str) -> p.Result[str]:
            """Look up a name."""
            return r[str].ok(request)

    class TypeCheckingOnlyService(s[bool]):
        """Operation whose request is imported only under TYPE_CHECKING."""

        def configure(self, request: FlextSettings) -> p.Result[bool]:
            """Configure from settings."""
            return r[bool].ok(bool(request))

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

    def test_operations_are_frozen_models(self) -> None:
        """A reported ``m.ServiceOperation`` rejects assignment: it is frozen."""
        operation = u.service_operations(self.CommandService)[0]
        field_name = next(iter(m.ServiceOperation.model_fields))

        with pytest.raises(m.ValidationError):
            setattr(operation, field_name, operation.summary)

    def test_discovery_is_cached_per_class(self) -> None:
        """A second call returns the same operations for the same class."""
        first = u.service_operations(self.CommandService)

        tm.that(u.service_operations(self.CommandService) is first, eq=True)

    def test_evaluated_annotations_are_accepted(self) -> None:
        """Annotations already evaluated to objects resolve the same way."""
        operations = u.service_operations(
            TestsFlextCoreServiceOperationsEvaluated.EvaluatedService
        )

        tm.that(
            [(op.name, op.request) for op in operations],
            eq=[("dispatch", m.Tests.DispatchRequest), ("status", None)],
        )

    def test_override_of_a_parent_operation_is_not_a_collision(self) -> None:
        """A subclass overriding its parent's operation keeps one operation."""
        operations = u.service_operations(self.OverridingService)

        tm.that(
            [(op.name, op.summary) for op in operations],
            eq=[("run", "Run the overriding side.")],
        )

    def test_malformed_service_class_is_created_without_error(self) -> None:
        """Discovery is lazy: class creation never checks operation shape."""
        tm.that(self.UndocumentedService().status().unwrap(), eq=True)

    # --- Failures --------------------------------------------------------

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
        self, service_type: type[s[bool]], operation: str, defect: str
    ) -> None:
        """Each malformed shape names the operation, module and fix."""
        with pytest.raises(TypeError) as raised:
            u.service_operations(service_type)

        message = str(raised.value)
        tm.that(message, has=f"{service_type.__qualname__}.{operation} ")
        tm.that(message, has=f"(module {__name__}, annotation ")
        tm.that(message.endswith(defect), eq=True)

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

    @pytest.mark.parametrize(
        ("service_type", "defect"),
        [
            (
                TestsFlextCoreServiceOperationsEvaluated.EvaluatedPlainReturnService,
                c.ERR_SERVICE_OPERATION_RESULT,
            ),
            (
                TestsFlextCoreServiceOperationsEvaluated.EvaluatedPlainRequestService,
                c.ERR_SERVICE_OPERATION_REQUEST,
            ),
        ],
    )
    def test_malformed_evaluated_annotation_raises(
        self, service_type: type[s[bool]], defect: str
    ) -> None:
        """Evaluated annotations are held to the same shape."""
        with pytest.raises(TypeError) as raised:
            u.service_operations(service_type)

        tm.that(str(raised.value).endswith(defect), eq=True)

    def test_sibling_name_collision_raises(self) -> None:
        """Two sibling classes declaring one operation name fail discovery."""
        with pytest.raises(TypeError) as raised:
            u.service_operations(self.CollidingService)

        owners = f"{self.ReadSide.__qualname__}, {self.WriteSide.__qualname__}"
        tm.that(
            str(raised.value).endswith(
                c.ERR_SERVICE_OPERATION_COLLISION.format(owners=owners)
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
                service=self.EmptyService.__qualname__, module=__name__
            ),
        )

