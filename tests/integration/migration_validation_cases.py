"""Additional migration validation cases kept outside the collected test module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import FlextContainer
from tests import r, t, tm, u

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests import p


def capture_output[T](
    emit: Callable[[], T], capture: p.Tests.OutputCapture, *, contains: str
) -> T:
    """Check native captured output without assuming a configured logging sink.

    Returns:
        The resulting ``T``.

    """
    result = emit()
    captured = capture.readouterr()
    tm.that(captured.out + captured.err, has=contains)
    return result


class TestsFlextFlextMigrationApplicationCase:
    """Exercise the public application composition contract."""

    @staticmethod
    def test_application_functionality_works(capfd: p.Tests.OutputCapture) -> None:
        """Verify application functionality works correctly."""

        class ApplicationExample:
            """Example application using r and logging."""

            def __init__(self) -> None:
                super().__init__()
                self.logger = u.fetch_logger(__name__)
                self.container = FlextContainer()

            def process_data(self, data: t.StrMapping) -> p.Result[t.JsonMapping]:
                """Typical data processing method.

                Returns:
                    The resulting ``p.Result[t.JsonMapping]``.

                """
                if not data:
                    return r[t.JsonMapping].fail("Data required")
                self.logger.info("Processing data", size=len(data))
                processed: t.JsonMapping = {"original": str(data), "processed": True}
                return r[t.JsonMapping].ok(processed)

        app = ApplicationExample()
        result = capture_output(
            lambda: app.process_data({"key": "value"}),
            capfd,
            contains="Processing data",
        )
        tm.ok(result.success)
        tm.that(result.value["processed"], eq=True)
