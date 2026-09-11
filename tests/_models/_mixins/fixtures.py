"""Fixture dictionary model helper namespace."""

from __future__ import annotations

from .fixture_payloads import TestsFlextModelsFixturePayloadsMixin
from .fixture_suite import TestsFlextModelsFixtureSuiteMixin


class TestsFlextModelsFixtureDictsMixin(
    TestsFlextModelsFixturePayloadsMixin, TestsFlextModelsFixtureSuiteMixin
):
    """Fixture dictionary model helpers."""


__all__: list[str] = ["TestsFlextModelsFixtureDictsMixin"]
