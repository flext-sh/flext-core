"""Basic test configuration for flext-core."""

from __future__ import annotations

import pytest


@pytest.fixture
def test_scenario():
    """Basic test scenario fixture."""
    return {"status": "test"}


class TestScenario:
    """Basic test scenario class."""

    def __init__(self, name: str = "test") -> None:
        self.name = name
        self.status = "active"
