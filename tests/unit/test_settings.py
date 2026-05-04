"""Behavior contract for flext_core.FlextSettings — public API only."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from flext_core import FlextSettings
from tests import t


@pytest.fixture(autouse=True)
def reset_flext_settings_singleton() -> Generator[None]:
    """Isolate singleton state across settings tests."""
    FlextSettings.reset_for_testing()
    try:
        yield
    finally:
        FlextSettings.reset_for_testing()


class TestsFlextSettings:
    """Coverage for the exported ``TestsFlextSettings`` test surface."""

    def test_reset_for_testing_drops_cached_singleton(self) -> None:
        first = FlextSettings.fetch_global()
        assert FlextSettings.fetch_global() is first

        FlextSettings.reset_for_testing()

        second = FlextSettings.fetch_global()
        assert second is not first


__all__: t.MutableSequenceOf[str] = ["TestsFlextSettings"]
