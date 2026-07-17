"""Additional migration validation cases kept outside the collected test module."""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

from flext_tests import r, tm

from flext_core import FlextContainer
from tests.typings import p, t
from tests.utilities import u

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.protocols import p


def capture_stdout[T](emit: Callable[[], T], *, contains: str) -> T:
    """Capture stdout until the expected observable message is emitted."""
    stream = io.StringIO()
    with redirect_stdout(stream):
        result = emit()
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline and contains not in stream.getvalue():
            time.sleep(0.01)
    tm.that(stream.getvalue(), has=contains)
    return result


class TestsFlextFlextMigrationApplicationCase:
    """Exercise the public application composition contract."""

    def test_application_functionality_works(self) -> None:
        """Verify application functionality works correctly."""

        class ApplicationExample:
            """Example application using r and logging."""

            def __init__(self) -> None:
                super().__init__()
                self.logger = u.fetch_logger(__name__)
                self.container = FlextContainer()

            def process_data(
                self,
                data: t.StrMapping,
            ) -> p.Result[t.JsonMapping]:
                """Typical data processing method."""
                if not data:
                    return r[t.JsonMapping].fail("Data required")
                self.logger.info("Processing data", size=len(data))
                processed: t.JsonMapping = {
                    "original": str(data),
                    "processed": True,
                }
                return r[t.JsonMapping].ok(processed)

        app = ApplicationExample()
        result = capture_stdout(
            lambda: app.process_data({"key": "value"}),
            contains="Processing data",
        )
        tm.that(result.success, eq=True)
        tm.that(result.value["processed"], eq=True)
