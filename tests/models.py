"""Models for flext-core tests.

Provides TestsFlextModels using composition with FlextTestsModels and FlextModels.
All generic test models come from flext_tests.

Architecture:
- FlextTestsModels (flext_tests) = Generic models for all FLEXT projects
- FlextModels (flext_core) = Core domain models
- TestsFlextModels (tests/) = flext-core-specific models using composition

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import override

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextModels, m, p, t


class TestsFlextModels:
    """Models for flext-core tests - uses composition with FlextTestsModels.

    Architecture: Uses composition (not inheritance) with FlextTestsModels and FlextModels
    for flext-core-specific model definitions.

    Access patterns:
    - TestsFlextModels.FlextTestsModels.Tests.* = flext_tests test models (via composition)
    - TestsFlextModels.Core.* = flext-core-specific test models
    - TestsFlextModels.FlextModels.Entity, .FlextModels.Value, etc. = FlextModels domain models (via composition)

    Rules:
    - Use composition, not inheritance (FlextTestsModels deprecates subclassing)
    - flext-core-specific models go in Core namespace
    - Generic models accessed via FlextTestsModels.Tests namespace
    """

    AggregateRoot = FlextModels.AggregateRoot
    DomainEvent = FlextModels.DomainEvent

    # Type aliases for domain test input
    type DomainInputValue = t.ContainerValue | p.HasModelDump
    type DomainInputMapping = Mapping[str, TestsFlextModels.DomainInputValue]
    type DomainExpectedResult = t.ContainerValue | type[t.ContainerValue]

    class Core:
        """flext-core-specific test models namespace."""

        class DomainTestEntity(m.Entity):
            """Test entity for domain tests."""

            model_config = ConfigDict(frozen=False)

            name: str
            value: t.ContainerValue

        class DomainTestValue(m.Value):
            """Test value object for domain tests."""

            model_config = ConfigDict(frozen=True)

            data: str = ""
            count: int

        class CustomEntity:
            """Custom entity with configurable ID attribute."""

            def __init__(self, custom_id: str | None = None) -> None:
                """Initialize custom entity with ID."""
                self.custom_id = custom_id

        class SimpleValue:
            """Simple value object without model_dump."""

            def __init__(self, data: str) -> None:
                """Initialize simple value object."""
                self.data = data

        class ComplexValue:
            """FlextModels.Value object with non-hashable attributes."""

            def __init__(self, data: str, items: list[str]) -> None:
                """Initialize complex value with non-hashable items."""
                self.data = data
                self.items = items  # list is not hashable

        class NoDict:
            """Object without __dict__, using __slots__."""

            __slots__ = ("value",)

            def __init__(self, value: int) -> None:
                """Initialize object without __dict__."""
                object.__setattr__(self, "value", value)

            @override
            def __repr__(self) -> str:
                """Return string representation."""
                return f"NoDict({getattr(self, 'value', None)})"

        class MutableObj:
            """Mutable object for immutability testing."""

            def __init__(self, value: int) -> None:
                """Initialize mutable object."""
                self.value = value

        class ImmutableObj:
            """Immutable object with custom __setattr__."""

            _frozen: bool = True

            def __init__(self, value: int) -> None:
                """Initialize immutable object."""
                object.__setattr__(self, "value", value)

            @override
            def __setattr__(self, name: str, value: object) -> None:
                """Prevent attribute setting if frozen."""
                if self._frozen:
                    msg = "Object is frozen"
                    raise AttributeError(msg)
                object.__setattr__(self, name, value)

        class NoConfigNoSetattr:
            """Object without model_config or __setattr__."""

        class NoSetattr:
            """Object without __setattr__."""

        # ParseOptions reference for string parser tests
        # ParseOptions reference for string parser tests
        class ParseOptions(FlextModels.CollectionsParseOptions):
            """Parse options - real inheritance."""

    class ParseDelimitedCase(BaseModel):
        """Test case for parse_delimited method."""

        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

        text: str
        delimiter: str
        expected: list[str] | None = None
        expected_error: str | None = None
        options: FlextModels.CollectionsParseOptions | None = None
        strip: bool = True
        remove_empty: bool = True
        validator: Callable[[str], bool] | None = None
        use_legacy: bool = False
        description: str = Field(default="", exclude=True)

    class SplitEscapeCase(BaseModel):
        """Test case for split_on_char_with_escape method."""

        model_config = ConfigDict(frozen=True)

        text: str
        split_char: str
        escape_char: str = "\\"
        expected: list[str] | None = None
        expected_error: str | None = None
        description: str = Field(default="", exclude=True)

    class NormalizeWhitespaceCase(BaseModel):
        """Test case for normalize_whitespace method."""

        model_config = ConfigDict(frozen=True)

        text: str
        pattern: str = r"\s+"
        replacement: str = " "
        expected: str | None = None
        expected_error: str | None = None
        description: str = Field(default="", exclude=True)

    class RegexPipelineCase(BaseModel):
        """Test case for apply_regex_pipeline method."""

        model_config = ConfigDict(frozen=True)

        text: str
        patterns: list[tuple[str, str] | tuple[str, str, int]]
        expected: str | None = None
        expected_error: str | None = None
        description: str = Field(default="", exclude=True)

    class ObjectKeyCase(BaseModel):
        """Test case for get_object_key method."""

        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

        obj: t.ContainerValue
        expected_contains: list[str] | None = None
        expected_exact: str | None = None
        description: str = Field(default="", exclude=True)


class AutomatedTestScenario(BaseModel):
    """Pydantic v2 model for automated test scenarios."""

    model_config = ConfigDict(frozen=True)

    description: str
    input: t.ConfigurationMapping
    expected_success: bool


__all__ = [
    "AutomatedTestScenario",
    "TestsFlextModels",
]
