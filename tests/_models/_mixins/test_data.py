"""Static test data model helper namespace."""

from __future__ import annotations

from tests._models._mixins.test_data_identity import (
    TestsFlextModelsTestDataIdentityMixin,
)
from tests._models._mixins.test_data_values import TestsFlextModelsTestDataValuesMixin


class TestsFlextModelsTestDataMixin(
    TestsFlextModelsTestDataValuesMixin,
    TestsFlextModelsTestDataIdentityMixin,
):
    """Static test data model helpers."""


__all__: list[str] = ["TestsFlextModelsTestDataMixin"]
