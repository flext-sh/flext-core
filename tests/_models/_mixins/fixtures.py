"""Fixture dictionary model helper namespace."""

from __future__ import annotations

from tests._models._mixins.fixture_payloads import TestsFlextModelsFixturePayloadsMixin
from tests._models._mixins.fixture_suite import TestsFlextModelsFixtureSuiteMixin


class TestsFlextModelsFixtureDictsMixin(
    TestsFlextModelsFixturePayloadsMixin, TestsFlextModelsFixtureSuiteMixin
):
    """Fixture dictionary model helpers."""


__all__: list[str] = ["TestsFlextModelsFixtureDictsMixin"]
