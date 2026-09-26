"""Services whose operation annotations are evaluated objects.

This module deliberately omits ``from __future__ import annotations`` so that
``inspect.get_annotations`` returns evaluated objects instead of strings, the
form a consumer module without the future import produces.
"""

from typing import override

from flext_tests import r

from tests.base import s
from tests.models import m
from tests.protocols import p


class TestsFlextCoreServiceOperationsEvaluated:
    """Services declared with evaluated (non-string) annotations."""

    class EvaluatedService(s[bool]):
        """Service with one request operation and one input-less operation."""

        @override
        def execute(self) -> p.Result[bool]:
            """Execute is part of the kernel, never an operation."""
            return r[bool].ok(True)

        def dispatch(self, request: m.Tests.DispatchRequest) -> p.Result[str]:
            """Dispatch one command by name."""
            return r[str].ok(request.command_name)

        def status(self) -> p.Result[bool]:
            """Report readiness."""
            return r[bool].ok(True)

    class EvaluatedPlainReturnService(s[bool]):
        """Service whose operation returns a plain value instead of p.Result."""

        def status(self) -> bool:
            """Report readiness without a result."""
            return True

    class EvaluatedPlainRequestService(s[bool]):
        """Service whose operation takes a non-model request."""

        def lookup(self, request: str) -> p.Result[str]:
            """Look up a name."""
            return r[str].ok(request)

