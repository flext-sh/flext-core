"""Guard mapper and event model helpers."""

from __future__ import annotations

from collections import UserDict, UserList
from typing import TYPE_CHECKING, Annotated, ClassVar, override

from flext_core import m
from tests.typings import p, t

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator


class TestsFlextModelsGuardsMapperMixin:
    """Guard mapper and event model helpers."""

    class GuardSampleModel(m.BaseModel):
        """Sample model for guard testing."""

        name: str = "test"

    class NoModelDump:
        """Object without model_dump — should fail is_pydantic_model."""

    class LoggerLike(m.BaseModel):
        """Partial logger-like object for testing rejection by logger protocol check.

        Extends BaseModel to satisfy GuardInput typing. Intentionally omits
        required Logger protocol methods (name, bind, new, unbind, etc.) so that
        matches_type(instance, 'logger') returns False.
        """

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(
            arbitrary_types_allowed=True,
        )

        def debug(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

        def info(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

        def warning(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

        def error(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

        def exception(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

    # --- from test_models_context_full_coverage.py ---

    class ModelWithNoCallableDump:
        """Model with non-callable model_dump attribute."""

        model_dump = "bad"

    # --- from test_utilities_mapper_full_coverage.py ---

    class PortModel(m.BaseModel):
        """Model with port/nested for mapper take/extract tests."""

        port: int = 0
        nested: Annotated[
            t.JsonMapping,
            m.Field(default_factory=dict),
        ]

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
            """Items method."""
            msg = "bad items"
            raise RuntimeError(msg)

    class BadIter(UserList[str]):
        """UserList that explodes on __iter__ for error-path testing."""

        @override
        def __iter__(self) -> Iterator[str]:
            """Raise RuntimeError when iterated (guard stub)."""
            msg = "bad iter"
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

    class UserUpdatedEvent(m.DomainEvent):
        """Domain event for user updates."""

        user_id: Annotated[str, m.Field(description="Identifier of the updated user.")]
        old_name: Annotated[str, m.Field(description="Previous user name.")]
        new_name: Annotated[str, m.Field(description="Updated user name.")]
        timestamp: Annotated[
            float,
            m.Field(description="POSIX timestamp when the event fired."),
        ]


__all__: list[str] = ["TestsFlextModelsGuardsMapperMixin"]
