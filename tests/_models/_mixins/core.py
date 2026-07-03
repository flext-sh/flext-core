"""Core shared model helper namespace."""

from __future__ import annotations

from tests._models._mixins.core_errors import TestsFlextModelsCoreErrorsMixin
from tests._models._mixins.core_public import TestsFlextModelsCorePublicMixin
from tests._models._mixins.core_state import TestsFlextModelsCoreStateMixin


class TestsFlextModelsCoreMixin(
    TestsFlextModelsCoreStateMixin,
    TestsFlextModelsCoreErrorsMixin,
    TestsFlextModelsCorePublicMixin,
):
    """Core shared model helpers."""


__all__: list[str] = ["TestsFlextModelsCoreMixin"]
