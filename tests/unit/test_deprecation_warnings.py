"""Deprecation warning smoke tests."""

from __future__ import annotations

from tests import r


class TestsFlextDeprecationWarnings:
    def test_failure_result_contains_message(self) -> None:
        result = r[str].fail("deprecated")
        assert result.failure
        assert result.error == "deprecated"

    def test_success_result_for_non_deprecated_path(self) -> None:
        result = r[str].ok("ok")
        assert result.success
