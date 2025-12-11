"""Type system foundation for FLEXT tests.

Provides FlextTestsTypes, extending FlextTypes with test-specific type definitions
for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, TypeVar

from pydantic import BaseModel

from flext_core import FlextTypes
from flext_core.result import r

# Note: p.Tests.* protocols are defined in flext_tests.protocols
# Use forward reference strings "FlextTestsProtocols" in type hints if needed

TTestResult = TypeVar("TTestResult")
TTestModel = TypeVar("TTestModel")
TTestService = TypeVar("TTestService")

# File content type for test operations
type FileContent = (
    str | bytes | t.ConfigurationMapping | Sequence[Sequence[str]] | BaseModel
)


class FlextTestsTypes(FlextTypes):
    """Type system foundation for FLEXT tests - extends FlextTypes.

    Architecture: Extends FlextTypes with test-specific type aliases and definitions.
    All base types from FlextTypes are available through inheritance.
    Uses specific, directed types instead of t.GeneralValueType where possible.

    This class serves as a library of support and base for all tests in the FLEXT
    workspace projects, without being directed to any specific project.
    """

    class Tests:
        """Test-specific type definitions namespace.

        All test-specific types organized under t.Tests.* pattern.
        Use specific types instead of t.GeneralValueType where possible.
        """

        # Reuse StringMapping from flext_core.typings - no duplication
        type ContainerPortMapping = t.StringMapping
        """Mapping of container port names to host port bindings."""

        # Reuse ConfigurationMapping from flext_core.typings - no duplication
        type ContainerConfigMapping = t.ConfigurationMapping
        """Mapping for container configuration data with specific value types."""

        # Reuse ConfigurationMapping from flext_core.typings - no duplication
        type DockerComposeServiceMapping = t.ConfigurationMapping
        """Mapping for docker-compose service configuration with specific types."""

        # Reuse ConfigurationMapping from flext_core.typings - no duplication
        type ContainerStateMapping = t.ConfigurationMapping
        """Mapping for container state information with specific value types."""

        # Reuse ConfigurationMapping from flext_core.typings - no duplication
        type TestDataMapping = t.ConfigurationMapping
        """Mapping for test data with specific value types."""

        # Reuse ConfigurationMapping from flext_core.typings - no duplication
        type TestConfigMapping = t.ConfigurationMapping
        """Mapping for test configuration with specific value types."""

        # Reuse t.GeneralValueType from flext_core.typings - no duplication
        type TestResultValue = t.GeneralValueType
        """Type for test result values with specific constraints."""

        # Note: Generic callables can't use module TypeVars in type aliases
        # Use Callable[..., T] directly with TypeVar T when needed

        class Docker:
            """Docker-specific type definitions with specific types."""

            # Reuse StringMapping from flext_core.typings - no duplication
            type ContainerPorts = t.StringMapping
            """Container port mappings (container_port -> host:port)."""

            type ContainerLabels = t.StringMapping
            """Container labels mapping."""

            type ContainerEnvironment = Sequence[str]
            """Container environment variables as sequence."""

            # Reuse ConfigurationMapping from flext_core.typings - no duplication
            type ComposeFileConfig = t.ConfigurationMapping
            """Docker compose file configuration structure with specific types."""

            # Reuse StringMapping from flext_core.typings - no duplication
            type VolumeMapping = t.StringMapping
            """Volume mappings (host_path -> container_path)."""

            # Reuse ConfigurationMapping from flext_core.typings - no duplication
            type NetworkMapping = t.ConfigurationMapping
            """Network configuration mapping with specific types."""

            type ContainerHealthStatus = str
            """Container health status type (healthy, unhealthy, starting, none)."""

            type ContainerHealthStatusLiteral = str  # Future: Literal health values
            """Type-safe literal for container health status."""

            # Reuse ConfigurationMapping from flext_core.typings - no duplication
            type ContainerOperationResult = t.ConfigurationMapping
            """Result type for container operations with specific fields."""

        class Test:
            """Test-specific type definitions."""

            # Reuse ConfigurationMapping from flext_core.typings - no duplication
            type TestCaseData = t.ConfigurationMapping
            """Test case data structure with specific value types."""

            # Reuse ConfigurationMapping from flext_core.typings - no duplication
            # Note: Path is included in t.GeneralValueType via object compatibility
            type TestFixtureData = t.ConfigurationMapping
            """Test fixture data structure with specific value types."""

            type TestAssertionResult = Mapping[str, str | bool | int | None]
            """Test assertion result structure."""

            # Reuse ConfigurationMapping from flext_core.typings - no duplication
            type TestExecutionContext = t.ConfigurationMapping
            """Test execution context with specific metadata types."""

        class Factory:
            """Factory-specific type definitions for test factories (tt).

            Provides comprehensive type aliases for factory operations following
            FLEXT patterns. All types use centralized definitions from flext_core.
            """

            # Kind literals for factory methods
            type ModelKind = Literal[
                "user",
                "config",
                "service",
                "entity",
                "value",
                "command",
                "query",
                "event",
            ]
            """Kind parameter for model() factory method."""

            type ResultKind = Literal["ok", "fail", "from_value"]
            """Kind parameter for res() factory method."""

            type OpKind = Literal[
                "simple",
                "add",
                "format",
                "error",
                "type_error",
                "result_ok",
                "result_fail",
            ]
            """Kind parameter for op() factory method."""

            type BatchKind = Literal["user", "config", "service"]
            """Kind parameter for batch() factory method."""

            # Pattern and collection types
            type BatchPattern = Sequence[bool]
            """Pattern for batch result creation (True=success, False=failure)."""

            type FactoryCallable[T] = Callable[[], T]
            """Factory function type that creates instances of T."""

            type TransformCallable[T] = Callable[[T], T]
            """Transform function that modifies instances of T."""

            type ValidateCallable[T] = Callable[[T], bool]
            """Validation predicate that checks instances of T."""

            type KeyFactory[K] = Callable[[int], K]
            """Key factory function that generates keys from index."""

            type ValueFactory[K, V] = Callable[[K], V]
            """Value factory function that generates values from keys."""

            # Model union types (for type-safe factory returns)
            # Uses BaseModel as the base - all factory models are Pydantic models
            type FactoryModel = BaseModel
            """Base type for all factory model types (Pydantic BaseModel)."""

            type FactoryModelList = list[FlextTestsTypes.Tests.Factory.FactoryModel]
            """List of factory models."""

            type FactoryModelDict = dict[
                str, FlextTestsTypes.Tests.Factory.FactoryModel
            ]
            """Dictionary of factory models keyed by string ID."""

            # Result types
            type FactoryResult[T] = r[T]
            """FlextResult wrapper for factory operations."""

            type FactoryResultList[T] = list[r[T]]
            """List of FlextResult instances."""

            # Collection factory types
            type ListSource[T] = (
                Sequence[T]
                | Callable[[], T]
                | Literal["user", "config", "service", "entity", "value"]
            )
            """Source type for list() factory method."""

            type DictSource[K, V] = (
                Mapping[K, V]
                | Callable[[], tuple[K, V]]
                | Literal["user", "config", "service", "entity", "value"]
            )
            """Source type for dict() factory method."""

            # Generic factory types
            # Reuse types from flext_core.typings - no duplication
            type GenericArgs = Sequence[t.GeneralValueType]
            """Positional arguments for generic type instantiation."""

            type GenericKwargs = t.ConfigurationMapping
            """Keyword arguments for generic type instantiation."""

        class Files:
            """File-specific type definitions for test file operations (tf)."""

            # Reuse ScalarValue from flext_core.typings - no duplication
            type ScalarValue = t.ScalarValue
            """Scalar values that can be serialized directly."""

            # Reuse JsonValue from flext_core.typings - no duplication
            type SerializableValue = t.JsonValue
            """Values that can be serialized to JSON/YAML."""

            # File operation type definitions moved to module level

            type ReadResult[T] = (
                T
                | dict[str, t.GeneralValueType]
                | list[str | dict[str, t.GeneralValueType]]
            )
            """Result type for file read operations with generic support."""

            type FormatLiteral = Literal["json", "yaml", "csv", "txt", "md", "auto"]
            """Literal type for file format specification in create/read operations."""

            type OperationLiteral = Literal[
                "create",
                "read",
                "delete",
                "compare",
                "info",
            ]
            """Literal type for batch operation specification."""

            type ErrorModeLiteral = Literal["stop", "skip", "collect"]
            """Error handling mode for batch operations.

            - stop: Stop at first error
            - skip: Skip failed operations, continue with remaining
            - collect: Collect all errors, return BatchResult with failures
            """

            type BatchFiles = Mapping[str, FileContent] | Sequence[FileContent]
            """Type for batch file operations - Mapping or Sequence of files."""

        class Builders:
            """Builder-specific type definitions for test data construction (tb).

            Provides centralized types for FlextTestsBuilders following FLEXT patterns.
            Use t.Tests.Builders.* for access.
            """

            type BuilderValue = (
                t.GeneralValueType
                | BaseModel
                | r[t.GeneralValueType]
                | list[t.GeneralValueType]
                | dict[str, t.GeneralValueType]
            )
            """Type for values that can be added to builder."""

            type BuilderDict = dict[str, FlextTestsTypes.Tests.Builders.BuilderValue]
            """Type for builder internal data structure."""

            # Reuse ConfigurationMapping from flext_core.typings - no duplication
            type BuilderMapping = t.ConfigurationMapping
            """Type for builder mappings."""

            type BuilderSequence = Sequence[FlextTestsTypes.Tests.Builders.BuilderValue]
            """Type for builder sequences."""

            type ParametrizedCase = tuple[
                str, FlextTestsTypes.Tests.Builders.BuilderDict
            ]
            """Type for parametrized test cases (test_id, data)."""

            # Reuse HandlerCallable from flext_core.typings - no duplication
            type TransformFunc = t.HandlerCallable
            """Type for transformation functions."""

            # Reuse ConditionCallable from flext_core.typings - no duplication
            type ValidateFunc = t.ConditionCallable
            """Type for validation functions."""

            type ResultBuilder[T] = Callable[[], r[T]]
            """Type for result builder functions that return FlextResult."""

            type ResultTransform[T, U] = Callable[[T], r[U]]
            """Type for result transformation functions."""

        class Matcher:
            """Matcher-specific type definitions for test assertions (tm.* methods).

            All types follow FLEXT patterns:
            - Use type aliases for complex types
            - Use Callable for predicates and validators
            - Use Mapping/Sequence for structured data
            - All types are documented with docstrings
            """

            # =====================================================================
            # Length and Size Specifications
            # =====================================================================

            type LengthSpec = int | tuple[int, int]
            """Length specification: exact int or (min, max) tuple.

            Examples:
                len=5              # Exact length 5
                len=(1, 10)        # Length between 1 and 10 (inclusive)
            """

            # =====================================================================
            # Deep Structural Matching
            # =====================================================================

            type DeepSpec = Mapping[str, Callable[[object], bool] | object]
            """Deep structural matching specification: path -> value or predicate.

            Supports unlimited nesting with dot notation paths.
            Values can be direct values or predicate functions.

            Examples:
                deep={"user.name": "John"}                    # Direct value
                deep={"user.email": lambda e: "@" in e}       # Predicate
                deep={"user.profile.age": lambda a: a >= 18}  # Deep nesting
            """

            type PathSpec = str | Sequence[str]
            """Path specification for nested value extraction.

            Supports dot notation (str) or sequence of keys (Sequence[str]).

            Examples:
                path="user.profile.name"        # Dot notation
                path=["user", "profile", "name"]  # Sequence of keys
            """

            # =====================================================================
            # Predicates and Validators
            # =====================================================================

            type PredicateSpec = Callable[[object], bool]
            """Custom predicate function for validation.

            Takes a value and returns True if validation passes.

            Examples:
                where=lambda x: x > 0
                where=lambda u: u.age >= 18 and u.verified
            """

            type ValueSpec = Callable[[object], bool] | object
            """Value specification: direct value or predicate function.

            Used in deep matching and custom validation.
            Can be a direct value for equality check or a predicate for custom logic.
            """

            type AssertionSpec = (
                Mapping[str, object]
                | Callable[[object], bool]
                | type[object]
                | tuple[type[object], ...]
            )
            """Assertion specification for flexible validation.

            Supports multiple assertion types:
            - Mapping: Key-value pairs for structured validation
            - Callable: Predicate function
            - type: Type check (single type)
            - tuple[type, ...]: Type check (multiple types)

            Note: object appears in union but is handled by type checking.
            """

            # =====================================================================
            # Containment Specifications
            # =====================================================================

            type ContainmentSpec = object | Sequence[object]
            """Containment specification: single item or sequence of items.

            Used for has/lacks parameters that check if container contains item(s).

            Examples:
                has="key"              # Single item
                has=["key1", "key2"]   # Multiple items
            """

            type ExclusionSpec = str | Sequence[str]
            """Exclusion specification: single string or sequence of strings.

            Used for lacks/excludes parameters that check if container does NOT contain.

            Examples:
                lacks="error"              # Single exclusion
                lacks=["error", "fail"]    # Multiple exclusions
            """

            # =====================================================================
            # Sequence Assertions
            # =====================================================================

            type SequencePredicate = type[object] | Callable[[object], bool]
            """Sequence predicate: type check or custom predicate.

            Used for all_/any_ parameters that validate sequence items.

            Examples:
                all_=str                    # All items are strings
                all_=lambda x: x > 0        # All items pass predicate
            """

            # Use object for runtime compatibility - protocol is only for type checking
            type SortKey = bool | Callable[[object], object]
            """Sort key specification: boolean or key function.

            Used for sorted parameter.
            - True: Check if sequence is sorted (ascending)
            - Callable: Key function returning a comparable type

            The key function must return a type that supports __lt__ (less-than
            comparison), as required by Python's sorted() function.

            Examples:
                sorted=True                  # Check ascending sort
                sorted=lambda x: x.id       # Check sort by id (int supports __lt__)
            """

            # =====================================================================
            # Mapping Assertions
            # =====================================================================

            type KeySpec = Sequence[str] | set[str]
            """Key specification: sequence or set of keys.

            Used for keys/lacks_keys parameters.

            Examples:
                keys=["id", "name"]         # Sequence
                keys={"id", "name"}         # Set
            """

            type KeyValueSpec = tuple[str, object] | Mapping[str, object]
            """Key-value specification: single pair or mapping.

            Used for kv parameter that validates key-value pairs.

            Examples:
                kv=("status", "active")                    # Single pair
                kv={"status": "active", "type": "user"}    # Multiple pairs
            """

            # =====================================================================
            # Object Assertions
            # =====================================================================

            type AttributeSpec = str | Sequence[str]
            """Attribute specification: single attribute or sequence.

            Used for attrs/methods parameters.

            Examples:
                attrs="name"                    # Single attribute
                attrs=["name", "email"]          # Multiple attributes
            """

            type AttributeValueSpec = tuple[str, object] | Mapping[str, object]
            """Attribute-value specification: single pair or mapping.

            Used for attr_eq parameter that validates attribute values.

            Examples:
                attr_eq=("status", "active")                    # Single pair
                attr_eq={"status": "active", "type": "user"}   # Multiple pairs
            """

            # =====================================================================
            # Error Validation
            # =====================================================================

            type ErrorCodeSpec = str | Sequence[str]
            """Error code specification: single code or sequence.

            Used for code/code_has parameters in tm.fail().

            Examples:
                code="VALIDATION"                    # Exact code
                code_has=["VALID", "ERROR"]          # Contains codes
            """

            # Reuse ConfigurationMapping from flext_core.typings - no duplication
            type ErrorDataSpec = t.ConfigurationMapping
            """Error data specification: key-value pairs.

            Used for data parameter in tm.fail() to validate error metadata.

            Examples:
                data={"field": "email", "reason": "invalid"}
            """

            # =====================================================================
            # Scope Configuration
            # =====================================================================

            type CleanupSpec = Sequence[Callable[[], None]]
            """Cleanup specification: sequence of cleanup functions.

            Used for cleanup parameter in tm.scope().

            Examples:
                cleanup=[lambda: resource.cleanup(), lambda: db.close()]
            """

            # Reuse StringMapping from flext_core.typings - no duplication
            type EnvironmentSpec = t.StringMapping
            """Environment specification: mapping of env var names to values.

            Used for env parameter in tm.scope().

            Examples:
                env={"API_KEY": "test", "DEBUG": "true"}
            """


t = FlextTestsTypes

__all__ = [
    "FlextTestsTypes",
    "TTestModel",
    "TTestResult",
    "TTestService",
    "t",
]
