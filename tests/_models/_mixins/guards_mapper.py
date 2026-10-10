"""Guard mapper and event model helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict
from typing import TYPE_CHECKING, Annotated, override

from typing_extensions import TypeForm

from flext_core import m, u
from tests import t

if TYPE_CHECKING:
    from collections.abc import ItemsView


class TestsFlextModelsGuardsMapperMixin:
    """Guard mapper and event model helpers."""

    # --- from test_models_context_full_coverage.py ---

    # --- from test_utilities_mapper_full_coverage.py ---

    class PortModel(m.BaseModel):
        """Model with port/nested for mapper take/extract tests."""

        port: int = 0
        nested: t.JsonMapping = m.Field(
            default_factory=u.empty(TypeForm(t.JsonMapping))
        )

    class MaybeModel(m.BaseModel):
        """Model with optional field for take tests."""

        x: str | None = None

    class GroupModel(m.BaseModel):
        """Model with optional kind for group tests."""

        kind: str | None = None

    class BadItems(UserDict[str, t.JsonValue]):
        """UserDict that explodes on items() for error-path testing."""

        @override
        def items(self) -> ItemsView[str, t.JsonValue]:
            """Items method.

            Raises:
                RuntimeError: If bad items.

            """
            msg = "bad items"
            raise RuntimeError(msg)

    # --- from test_architectural_patterns.py ---

    class UserCreatedEvent(m.DomainEvent):
        """Domain event for user creation using FlextModels foundation."""

        user_id: Annotated[str, m.Field(description="Identifier of the created user.")]
        user_name: Annotated[str, m.Field(description="Name assigned to the new user.")]
        timestamp: Annotated[
            float,
            m.Field(description="POSIX timestamp when the event fired."),
        ]


__all__: list[str] = ["TestsFlextModelsGuardsMapperMixin"]
