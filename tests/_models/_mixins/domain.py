"""Domain and parser model helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, override

from flext_core import m

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.typings import p, t


class TestsFlextModelsDomainMixin:
    """Domain and parser model helpers."""

    class EmailResponse(m.BaseModel):
        """Shared email response model for tests."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        status: str
        message_id: str

    class DomainTestEntity(m.Entity):
        """Test entity for domain tests."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=False)

        name: Annotated[str, m.Field(description="Entity display name.")]
        value: Annotated[t.JsonValue, m.Field(description="Entity payload value.")]

    class DomainTestValue(m.Value):
        """Test value object for domain tests."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        data: Annotated[str, m.Field(description="Value payload string.")] = ""
        count: Annotated[int, m.Field(description="Occurrence counter.")]

    class CustomEntity(m.BaseModel):
        """Custom entity with configurable ID attribute."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=False)

        custom_id: str | None = None

        def __init__(self, custom_id: str | None = None, **kwargs: t.Scalar) -> None:
            """Initialize custom entity with ID."""
            super().__init__(custom_id=custom_id, **kwargs)

    class SimpleValue(m.BaseModel):
        """Simple value object — tests behavior when model_dump is absent at runtime."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=False)

        data: str = ""

        def __init__(self, data: str = "", **kwargs: t.Scalar) -> None:
            """Initialize simple value object."""
            super().__init__(data=data, **kwargs)

    class ComplexValue(m.BaseModel):
        """Value object with non-hashable attributes."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=False)

        data: str = ""
        items: Annotated[t.StrSequence, m.Field(default_factory=list)]

        def __init__(
            self, data: str = "", items: t.StrSequence | None = None, **kwargs: t.Scalar
        ) -> None:
            """Initialize complex value with non-hashable items."""
            super().__init__(data=data, items=items or [], **kwargs)

    class NoDict(m.BaseModel):
        """Model for testing value-comparison fallback paths in domain utilities."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=False)

        value: int = 0

        def __init__(self, value: int = 0, **kwargs: t.Scalar) -> None:
            """Initialize model for domain utility edge-case testing."""
            super().__init__(value=value, **kwargs)

        @override
        def __repr__(self) -> str:
            """Return string representation."""
            return f"NoDict(value={self.value})"

    class MutableObj:
        """Mutable t.JsonValue for immutability testing."""

        def __init__(self, value: int) -> None:
            """Initialize mutable t.JsonValue."""
            self.value = value

    class NoSettingsNoSetattr:
        """Object without model_config or __setattr__."""

    class NoSetattr:
        """Object without __setattr__."""

    class ParseOptions(m.BaseModel):
        """Test-local parse options after production model removal."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        strip: bool = True
        remove_empty: bool = True
        validator: Callable[[str], bool] | None = None

    class ParseDelimitedCase(m.BaseModel):
        """Test case for parse_delimited method."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(
            frozen=True, arbitrary_types_allowed=True
        )

        text: str
        delimiter: str
        expected: t.StrSequence | None = None
        expected_error: str | None = None
        options: p.BaseModel | None = None
        strip: bool = True
        remove_empty: bool = True
        validator: Callable[[str], bool] | None = None
        use_legacy: bool = False
        description: Annotated[str, m.Field(exclude=True)] = ""

    class SplitEscapeCase(m.BaseModel):
        """Test case for split_on_char_with_escape method."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        text: str
        split_char: str
        escape_char: str = "\\"
        expected: t.StrSequence | None = None
        expected_error: str | None = None
        description: Annotated[str, m.Field(exclude=True)] = ""

    class NormalizeWhitespaceCase(m.BaseModel):
        """Test case for normalize_whitespace method."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        text: str
        pattern: str = r"\s+"
        replacement: str = " "
        expected: str | None = None
        expected_error: str | None = None
        description: Annotated[str, m.Field(exclude=True)] = ""

    class RegexPipelineCase(m.BaseModel):
        """Test case for apply_regex_pipeline method."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        text: str
        patterns: t.SequenceOf[tuple[str, str] | tuple[str, str, int]]
        expected: str | None = None
        expected_error: str | None = None
        description: Annotated[str, m.Field(exclude=True)] = ""

    class ObjectKeyCase(m.BaseModel):
        """Test case for get_object_key method."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(
            frozen=True, arbitrary_types_allowed=True
        )

        obj: t.JsonValue
        expected_contains: t.StrSequence | None = None
        expected_exact: str | None = None
        description: Annotated[str, m.Field(exclude=True)] = ""


__all__: list[str] = ["TestsFlextModelsDomainMixin"]
