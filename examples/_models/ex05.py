"""Example 05 mixins models.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from examples import c
from flext_core import m, p


class ExamplesFlextModelsEx05:
    """Example 05 mixins model namespace."""

    class UserModel(m.Value):
        name: str = m.Field(description="User display name")
        status: c.Examples.StatusEnum = m.Field(
            description="User account status",
        )
        age: int = m.Field(description="User age in years")

        @m.field_validator("status", mode="before")
        @classmethod
        def normalize_status(
            cls,
            value: str | c.Examples.StatusEnum,
        ) -> c.Examples.StatusEnum:
            if isinstance(value, c.Examples.StatusEnum):
                return value
            return c.Examples.StatusEnum(value)

    class HandlerBad(m.Value):
        """Marker model that is not a valid handler."""

    class HandlerLike(m.Value):
        data: p.ConfigMap = m.Field(
            default_factory=lambda: m.ConfigMap(root={}),
            description="Handler payload map",
        )

    class GoodProcessor(m.Value):
        @staticmethod
        def process() -> bool:
            return True

        @override
        @classmethod
        def validate(cls, value: p.ConfigMap) -> ExamplesFlextModelsEx05.GoodProcessor:
            return cls.model_validate(value)

    class BadProcessor(m.Value):
        """Marker model that is not a valid processor."""


__all__: list[str] = ["ExamplesFlextModelsEx05"]
