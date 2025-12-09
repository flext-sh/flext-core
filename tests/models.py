"""Models for flext-core tests.

Provides TestsFlextModels, extending FlextTestsModels with flext-core-specific models.
All generic test models come from flext_tests.

Architecture:
- FlextTestsModels (flext_tests) = Generic models for all FLEXT projects
- TestsFlextModels (tests/) = flext-core-specific models extending FlextTestsModels

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from flext_core import FlextModels, FlextProtocols, FlextTypes
from flext_tests.models import FlextTestsModels


class TestsFlextModels(FlextTestsModels):
    """Models for flext-core tests - extends FlextTestsModels.

    Architecture: Extends FlextTestsModels with flext-core-specific model definitions.
    All generic models from FlextTestsModels are available through inheritance.

    Rules:
    - NEVER redeclare models from FlextTestsModels
    - Only flext-core-specific models allowed
    - All generic models come from FlextTestsModels
    """

    # NOTE: FlextTestsModels extends FlextModels and exposes:
    # - Entity, Value, AggregateRoot, DomainEvent, Collections
    # - Docker namespace with ContainerInfo, ContainerConfig, ContainerState, ComposeConfig
    # These are available through inheritance.

    # Type aliases for domain test input
    type DomainInputValue = (
        FlextTypes.GeneralValueType | FlextProtocols.HasModelDump | object
    )
    type DomainInputMapping = Mapping[str, DomainInputValue]
    type DomainExpectedResult = (
        FlextTypes.GeneralValueType | type[FlextTypes.GeneralValueType]
    )

    class DomainTestEntity(FlextModels.Entity):
        """Test entity for domain tests."""

        name: str
        value: int

    class DomainTestValue(FlextModels.Value):
        """Test value object for domain tests."""

        data: str
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
        """Value object with non-hashable attributes."""

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
    class ParseOptions(FlextModels.Collections.ParseOptions):
        """Parse options - real inheritance."""

    @dataclass(frozen=True, slots=True)
    class ParseDelimitedCase:
        """Test case for parse_delimited method."""

        text: str
        delimiter: str
        expected: list[str] | None = None
        expected_error: str | None = None
        options: FlextModels.Collections.ParseOptions | None = None
        strip: bool = True
        remove_empty: bool = True
        validator: Callable[[str], bool] | None = None
        use_legacy: bool = False
        description: str = field(default="", compare=False)

    @dataclass(frozen=True, slots=True)
    class SplitEscapeCase:
        """Test case for split_on_char_with_escape method."""

        text: str
        split_char: str
        escape_char: str = "\\"
        expected: list[str] | None = None
        expected_error: str | None = None
        description: str = field(default="", compare=False)

    @dataclass(frozen=True, slots=True)
    class NormalizeWhitespaceCase:
        """Test case for normalize_whitespace method."""

        text: str
        pattern: str = r"\s+"
        replacement: str = " "
        expected: str | None = None
        expected_error: str | None = None
        description: str = field(default="", compare=False)

    @dataclass(frozen=True, slots=True)
    class RegexPipelineCase:
        """Test case for apply_regex_pipeline method."""

        text: str
        patterns: list[tuple[str, str] | tuple[str, str, int]]
        expected: str | None = None
        expected_error: str | None = None
        description: str = field(default="", compare=False)

    @dataclass(frozen=True, slots=True)
    class ObjectKeyCase:
        """Test case for get_object_key method."""

        obj: object
        expected_contains: list[str] | None = None
        expected_exact: str | None = None
        description: str = field(default="", compare=False)


__all__ = [
    "TestsFlextModels",
]
