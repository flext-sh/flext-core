"""Dispatcher test helpers for flext-core tests."""

from __future__ import annotations

from typing import override

from flext_tests import h, r

from tests.protocols import p
from tests.typings import t


class TestsFlextUtilitiesDispatchMixin:
    """Dispatcher test helpers."""

    class Handler(h[t.JsonPayload, t.JsonPayload]):
        """Simple handler used by public registry scenarios."""

        @override
        def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok(message)


__all__: list[str] = ["TestsFlextUtilitiesDispatchMixin"]
