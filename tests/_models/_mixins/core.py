"""Core shared model helper namespace."""

from __future__ import annotations

from .core_errors import TestsFlextModelsCoreErrorsMixin
from .core_public import TestsFlextModelsCorePublicMixin
from .core_state import TestsFlextModelsCoreStateMixin


class TestsFlextModelsCoreMixin(
    TestsFlextModelsCoreStateMixin,
    TestsFlextModelsCoreErrorsMixin,
    TestsFlextModelsCorePublicMixin,
):
    """Core shared model helpers."""


__all__: list[str] = ["TestsFlextModelsCoreMixin"]
