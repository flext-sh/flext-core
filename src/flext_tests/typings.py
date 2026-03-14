"""Type system foundation for FLEXT tests.

Provides FlextTestsTypes, extending t with test-specific type definitions
for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAliasType, TypeGuard

from pydantic import BaseModel, InstanceOf

from flext_core import FlextTypes, m, r

type _Testobject = (
    FlextTypes.Primitives
    | None
    | bytes
    | datetime
    | Path
    | BaseModel
    | Sequence[_Testobject]
    | Mapping[str, _Testobject]
)


class FlextTestsTypes(FlextTypes):
    """Type system foundation for FLEXT tests - extends FlextTypes.

    Architecture: Extends FlextTypes with test-specific type aliases and definitions.
    All base types from FlextTypes are available through inheritance.
    Uses specific, directed types instead of FTestobject where possible.

    This class serves as a library of support and base for all tests in the FLEXT
    workspace projects, without being directed to any specific project.
    """

    class Tests:
        """Test-specific type definitions namespace.

        All test-specific types organized under t.Tests.* pattern.
        Use specific types instead of Testobject where possible.
        """

        type TestPayloadScalar = t.Primitives | None
        type Testobject = _Testobject
        type FileContent = (
            str
            | bytes
            | Mapping[str, _Testobject]
            | Sequence[Sequence[str]]
            | InstanceOf[BaseModel]
        )
        type ContainerPortMapping = Mapping[str, str]
        "Mapping of container port names to host port bindings."
        type ContainerConfigMapping = m.ConfigMap
        "Mapping for container configuration data with specific value types."
        type DockerComposeServiceMapping = m.ConfigMap
        "Mapping for docker-compose service configuration with specific types."
        type ContainerStateMapping = m.ConfigMap
        "Mapping for container state information with specific value types."
        type TestDataMapping = m.ConfigMap
        "Mapping for test data with specific value types."
        type TestConfigMapping = m.ConfigMap
        "Mapping for test configuration with specific value types."
        type object = _Testobject
        "Canonical payload value for test modules."
        type TestResultValue = _Testobject
        "Type for test result values with specific constraints."

        class Docker:
            """Docker-specific type definitions with specific types."""

            type ContainerPorts = Mapping[str, str]
            "Container port mappings (container_port -> host:port)."
            type ContainerLabels = Mapping[str, str]
            "Container labels mapping."
            type ContainerEnvironment = Sequence[str]
            "Container environment variables as sequence."
            type ComposeFileConfig = m.ConfigMap
            "Docker compose file configuration structure with specific types."
            type VolumeMapping = Mapping[str, str]
            "Volume mappings (host_path -> container_path)."
            type NetworkMapping = m.ConfigMap
            "Network configuration mapping with specific types."
            type ContainerHealthStatus = str
            "Container health status type (healthy, unhealthy, starting, none)."
            type ContainerHealthStatusLiteral = str
            "Type-safe literal for container health status."
            type ContainerOperationResult = m.ConfigMap
            "Result type for container operations with specific fields."

        class Test:
            """Test-specific type definitions."""

            type TestCaseData = m.ConfigMap
            "Test case data structure with specific value types."
            type TestFixtureData = m.ConfigMap
            "Test fixture data structure with specific value types."
            type TestAssertionResult = Mapping[str, str | bool | int | None]
            "Test assertion result structure."
            type TestExecutionContext = m.ConfigMap
            "Test execution context with specific metadata types."

        class Factory:
            """Factory-specific type definitions for test factories (tt).

            Provides comprehensive type aliases for factory operations following
            FLEXT patterns. All types use centralized definitions from flext_core.
            """

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
            "Kind parameter for model() factory method."
            type ResultKind = Literal["ok", "fail", "from_value"]
            "Kind parameter for res() factory method."
            type OpKind = Literal[
                "simple",
                "add",
                "format",
                "error",
                "type_error",
                "result_ok",
                "result_fail",
            ]
            "Kind parameter for op() factory method."
            type BatchKind = Literal["user", "config", "service"]
            "Kind parameter for batch() factory method."
            type BatchPattern = Sequence[bool]
            "Pattern for batch result creation (True=success, False=failure)."
            type FactoryCallable[T] = Callable[[], T]
            "Factory function type that creates instances of T."
            type TransformCallable[T] = Callable[[T], T]
            "Transform function that modifies instances of T."
            type ValidateCallable[T] = Callable[[T], bool]
            "Validation predicate that checks instances of T."
            type KeyFactory[K] = Callable[[int], K]
            "Key factory function that generates keys from index."
            type ValueFactory[K, V] = Callable[[K], V]
            "Value factory function that generates values from keys."
            type FactoryModel = BaseModel
            "Base type for all factory model types (Pydantic BaseModel)."
            type FactoryModelList = list[FlextTestsTypes.Tests.Factory.FactoryModel]
            "List of factory models."
            type FactoryModelDict = Mapping[
                str, FlextTestsTypes.Tests.Factory.FactoryModel
            ]
            "Dictionary of factory models keyed by string ID."
            type FactoryResult[T] = r[T]
            "r wrapper for factory operations."
            type FactoryResultList[T] = list[r[T]]
            "List of r instances."
            type ListSource[T] = (
                Sequence[T]
                | Callable[[], T]
                | Literal["user", "config", "service", "entity", "value"]
            )
            "Source type for list() factory method."
            type DictSource[K, V] = (
                Mapping[K, V]
                | Callable[[], tuple[K, V]]
                | Literal["user", "config", "service", "entity", "value"]
            )
            "Source type for dict() factory method."
            type GenericArgs = Sequence[FlextTestsTypes.Tests.Testobject]
            "Positional arguments for generic type instantiation."
            type GenericKwargs = m.ConfigMap
            "Keyword arguments for generic type instantiation."

        class Files:
            """File-specific type definitions for test file operations (tf)."""

            type ScalarValue = t.Scalar
            "Scalar values that can be serialized directly."
            type SerializableValue = _Testobject
            "Values that can be serialized to JSON/YAML."
            type ReadResult[T] = (
                T
                | Mapping[str, FlextTestsTypes.Tests.Testobject]
                | list[str | Mapping[str, FlextTestsTypes.Tests.Testobject]]
            )
            "Result type for file read operations with generic support."
            type FormatLiteral = Literal["json", "yaml", "csv", "txt", "md", "auto"]
            "Literal type for file format specification in create/read operations."
            type OperationLiteral = Literal[
                "create", "read", "delete", "compare", "info"
            ]
            "Literal type for batch operation specification."
            type ErrorModeLiteral = Literal["stop", "skip", "collect"]
            "Error handling mode for batch operations.\n\n            - stop: Stop at first error\n            - skip: Skip failed operations, continue with remaining\n            - collect: Collect all errors, return BatchResult with failures\n            "
            type BatchFiles = Mapping[str, t.Tests.object] | Sequence[t.Tests.object]
            "Type for batch file operations - Mapping or Sequence of files."

        class Builders:
            """Builder-specific type definitions for test data construction (tb).

            Provides centralized types for FlextTestsBuilders following FLEXT patterns.
            Use t.Tests.Builders.* for access.

            Uses FlextTestsTypes.Tests.Testobject as base since it already handles nested structures
            through Sequence payloads and Mapping payloads.
            r types are added on top for builder-specific needs.
            """

            type BuilderValue = FlextTestsTypes.Tests.Testobject
            "Type for values stored in builder."
            type BuilderDict = MutableMapping[str, FlextTestsTypes.Tests.Testobject]
            "Type for builder internal data structure."
            type BuilderOutputDict = Mapping[
                str,
                FlextTestsTypes.Tests.Testobject
                | r[FlextTestsTypes.Tests.Testobject]
                | list[
                    FlextTestsTypes.Tests.Testobject
                    | r[FlextTestsTypes.Tests.Testobject]
                ]
                | Mapping[str, FlextTestsTypes.Tests.Testobject],
            ]
            "Type for builder output dict after batch result conversion."
            type BuildOutputValue = (
                FlextTestsTypes.Tests.Testobject
                | r[FlextTestsTypes.Tests.Testobject]
                | list[
                    FlextTestsTypes.Tests.Testobject
                    | r[FlextTestsTypes.Tests.Testobject]
                ]
                | Mapping[str, FlextTestsTypes.Tests.Builders.BuildOutputValue]
            )
            "Type for build() output values, including r-wrapped results."
            type BuilderMapping = m.ConfigMap
            "Type for builder mappings."
            type BuilderSequence = Sequence[FlextTestsTypes.Tests.Testobject]
            "Type for builder sequences."
            type ParametrizedCase = tuple[
                str, Mapping[str, FlextTestsTypes.Tests.Testobject]
            ]
            "Type for parametrized test cases (test_id, data)."
            type TransformFunc = Callable[
                [FlextTestsTypes.Tests.Testobject],
                FlextTestsTypes.Tests.Testobject,
            ]
            "Type for transformation functions."
            type ValidateFunc = Callable[[FlextTestsTypes.Tests.Testobject], bool]
            "Type for validation functions."
            type ResultBuilder[T] = Callable[[], r[T]]
            "Type for result builder functions that return r."
            type ResultTransform[T, U] = Callable[[T], r[U]]
            "Type for result transformation functions."

        class Matcher:
            """Matcher-specific type definitions for test assertions (tm.* methods).

            All types follow FLEXT patterns:
            - Use type aliases for complex types
            - Use Callable for predicates and validators
            - Use Mapping/Sequence for structured data
            - All types are documented with docstrings
            """

            type MatcherKwargValue = (
                _Testobject
                | type
                | tuple[type, ...]
                | TypeAliasType
                | set[_Testobject]
                | Callable[..., _Testobject]
                | Mapping[str, Callable[..., _Testobject] | _Testobject]
            )
            "Union of all value types accepted by matcher kwargs.\n\n            Covers all field types across OkParams, FailParams, and ThatParams:\n            - _Testobject: scalar, bytes, BaseModel, Sequence, Mapping\n            - type/tuple[type, ...]: is_, not_ (type checking)\n            - TypeAliasType: eq, ne in OkParams\n            - set: keys, lacks_keys, eq (set values)\n            - Callable: where, all_, any_, sorted (predicates/key functions)\n            - Mapping[str, Callable | Value]: deep (structural matching)\n            "
            type LengthSpec = int | tuple[int, int]
            "Length specification: exact int or (min, max) tuple.\n\n            Examples:\n                len=5              # Exact length 5\n                len=(1, 10)        # Length between 1 and 10 (inclusive)\n            "
            type DeepSpec = Mapping[
                str,
                Callable[[FlextTestsTypes.Tests.Testobject], bool]
                | FlextTestsTypes.Tests.Testobject,
            ]
            'Deep structural matching specification: path -> value or predicate.\n\n            Supports unlimited nesting with dot notation paths.\n            Values can be direct values or predicate functions.\n\n            Examples:\n                deep={"user.name": "John"}                    # Direct value\n                deep={"user.email": lambda e: "@" in e}       # Predicate\n                deep={"user.profile.age": lambda a: a >= 18}  # Deep nesting\n            '
            type PathSpec = str | Sequence[str]
            'Path specification for nested value extraction.\n\n            Supports dot notation (str) or sequence of keys (Sequence[str]).\n\n            Examples:\n                path="user.profile.name"        # Dot notation\n                path=["user", "profile", "name"]  # Sequence of keys\n            '
            type PredicateSpec = Callable[[FlextTestsTypes.Tests.Testobject], bool]
            "Custom predicate function for validation.\n\n            Takes a value and returns True if validation passes.\n\n            Examples:\n                where=lambda x: x > 0\n                where=lambda u: u.age >= 18 and u.verified\n            "
            type ValueSpec = (
                Callable[[FlextTestsTypes.Tests.Testobject], bool]
                | FlextTestsTypes.Tests.Testobject
            )
            "Value specification: direct value or predicate function.\n\n            Used in deep matching and custom validation.\n            Can be a direct value for equality check or a predicate for custom logic.\n            "
            type AssertionSpec = (
                Mapping[str, FlextTestsTypes.Tests.Testobject]
                | Callable[[FlextTestsTypes.Tests.Testobject], bool]
                | type
                | tuple[type, ...]
            )
            "Assertion specification for flexible validation.\n\n            Supports multiple assertion types:\n            - Mapping: Key-value pairs for structured validation\n            - Callable: Predicate function\n            - type: Type check (single type)\n            - tuple[type, ...]: Type check (multiple types)\n            "
            type ContainmentSpec = (
                FlextTestsTypes.Tests.Testobject
                | Sequence[FlextTestsTypes.Tests.Testobject]
            )
            'Containment specification: single item or sequence of items.\n\n            Used for has/lacks parameters that check if container contains item(s).\n\n            Examples:\n                has="key"              # Single item\n                has=["key1", "key2"]   # Multiple items\n            '
            type ExclusionSpec = str | Sequence[str]
            'Exclusion specification: single string or sequence of strings.\n\n            Used for lacks/excludes parameters that check if container does NOT contain.\n\n            Examples:\n                lacks="error"              # Single exclusion\n                lacks=["error", "fail"]    # Multiple exclusions\n            '
            type SequencePredicate = (
                type | Callable[[FlextTestsTypes.Tests.Testobject], bool]
            )
            "Sequence predicate: type check or custom predicate.\n\n            Used for all_/any_ parameters that validate sequence items.\n\n            Examples:\n                all_=str                    # All items are strings\n                all_=lambda x: x > 0        # All items pass predicate\n            "
            type SortKey = (
                bool
                | Callable[
                    [FlextTestsTypes.Tests.Testobject],
                    FlextTestsTypes.Tests.Testobject,
                ]
            )
            "Sort key specification: boolean or key function.\n\n            Used for sorted parameter.\n            - True: Check if sequence is sorted (ascending)\n            - Callable: Key function returning a comparable type\n\n            The key function must return a type that supports __lt__ (less-than\n            comparison), as required by Python's sorted() function.\n\n            Examples:\n                sorted=True                  # Check ascending sort\n                sorted=lambda x: x.id       # Check sort by id (int supports __lt__)\n            "
            type KeySpec = Sequence[str] | set[str]
            'Key specification: sequence or set of keys.\n\n            Used for keys/lacks_keys parameters.\n\n            Examples:\n                keys=["id", "name"]         # Sequence\n                keys={"id", "name"}         # Set\n            '
            type KeyValueSpec = (
                tuple[str, FlextTestsTypes.Tests.Testobject]
                | Mapping[str, FlextTestsTypes.Tests.Testobject]
            )
            'Key-value specification: single pair or mapping.\n\n            Used for kv parameter that validates key-value pairs.\n\n            Examples:\n                kv=("status", "active")                    # Single pair\n                kv={"status": "active", "type": "user"}    # Multiple pairs\n            '
            type AttributeSpec = str | Sequence[str]
            'Attribute specification: single attribute or sequence.\n\n            Used for attrs/methods parameters.\n\n            Examples:\n                attrs="name"                    # Single attribute\n                attrs=["name", "email"]          # Multiple attributes\n            '
            type AttributeValueSpec = (
                tuple[str, FlextTestsTypes.Tests.Testobject]
                | Mapping[str, FlextTestsTypes.Tests.Testobject]
            )
            'Attribute-value specification: single pair or mapping.\n\n            Used for attr_eq parameter that validates attribute values.\n\n            Examples:\n                attr_eq=("status", "active")                    # Single pair\n                attr_eq={"status": "active", "type": "user"}   # Multiple pairs\n            '
            type ErrorCodeSpec = str | Sequence[str]
            'Error code specification: single code or sequence.\n\n            Used for code/code_has parameters in tm.fail().\n\n            Examples:\n                code="VALIDATION"                    # Exact code\n                code_has=["VALID", "ERROR"]          # Contains codes\n            '
            type ErrorDataSpec = m.ConfigMap
            'Error data specification: key-value pairs.\n\n            Used for data parameter in tm.fail() to validate error metadata.\n\n            Examples:\n                data={"field": "email", "reason": "invalid"}\n            '
            type CleanupSpec = Sequence[Callable[[], None]]
            "Cleanup specification: sequence of cleanup functions.\n\n            Used for cleanup parameter in tm.scope().\n\n            Examples:\n                cleanup=[lambda: resource.cleanup(), lambda: db.close()]\n            "
            type EnvironmentSpec = Mapping[str, str]
            'Environment specification: mapping of env var names to values.\n\n            Used for env parameter in tm.scope().\n\n            Examples:\n                env={"API_KEY": "test", "DEBUG": "true"}\n            '

    class Guards:
        """TypeGuard functions for type narrowing.

        Provides static methods for safe type narrowing in test builders,
        factories, and matchers. Use these guards for proper
        type safety with Python 3.13+.
        """

        @staticmethod
        def is_builder_dict(
            value: FlextTestsTypes.Tests.Testobject,
        ) -> TypeGuard[FlextTestsTypes.Tests.Builders.BuilderDict]:
            """Check if value is a BuilderDict (dict with str keys)."""
            return isinstance(value, dict)

        @staticmethod
        def is_builder_value(
            value: FlextTestsTypes.Tests.Testobject | type,
        ) -> TypeGuard[FlextTestsTypes.Tests.Testobject]:
            """Check if value is a valid BuilderValue."""
            if value is None:
                return True
            if isinstance(value, (str, int, float, bool, bytes)):
                return True
            if isinstance(value, BaseModel):
                return True
            return isinstance(value, (list, dict))

        @staticmethod
        def is_configuration_dict(
            value: FlextTestsTypes.Tests.Testobject,
        ) -> TypeGuard[Mapping[str, FlextTestsTypes.Tests.Testobject]]:
            """Check if value is a ConfigurationDict."""
            return isinstance(value, dict)

        @staticmethod
        def is_configuration_mapping(
            value: FlextTestsTypes.Tests.Testobject,
        ) -> TypeGuard[m.ConfigMap]:
            """Check if value is a ConfigurationMapping."""
            return isinstance(value, Mapping)

        @staticmethod
        def is_flext_result(
            value: FlextTestsTypes.Tests.Testobject,
        ) -> TypeGuard[r[FlextTestsTypes.Tests.Testobject]]:
            """Check if value is a r."""
            return r in type(value).__mro__

        @staticmethod
        def is_general_value(
            value: FlextTestsTypes.Tests.Testobject,
        ) -> TypeGuard[FlextTestsTypes.Tests.Testobject]:
            """Check if value is payload-compatible."""
            if value is None:
                return True
            if isinstance(value, (str, int, float, bool, bytes)):
                return True
            if BaseModel in type(value).__mro__:
                return True
            return isinstance(value, (list, dict))

        @staticmethod
        def is_mapping(
            value: FlextTestsTypes.Tests.Testobject,
        ) -> TypeGuard[Mapping[str, FlextTestsTypes.Tests.Testobject]]:
            """Check if value is a payload mapping."""
            return isinstance(value, dict)

        @staticmethod
        def is_model_kind(
            value: str,
        ) -> TypeGuard[Literal["user", "config", "service", "entity", "value"]]:
            """Check if value is a valid model kind literal."""
            return value in {"user", "config", "service", "entity", "value"}

        @staticmethod
        def is_sequence(
            value: FlextTestsTypes.Tests.Testobject,
        ) -> TypeGuard[Sequence[FlextTestsTypes.Tests.Testobject]]:
            """Check if value is a payload sequence."""
            return isinstance(value, (list, tuple)) and (
                not isinstance(value, (str, bytes))
            )

        @staticmethod
        def is_test_result_value(value: _Testobject) -> bool:
            """Check if value is a valid TestResultValue."""
            if value is None:
                return True
            if isinstance(value, (str, int, float, bool)):
                return True
            if isinstance(value, (list, tuple)):
                return True
            return isinstance(value, dict)


t = FlextTestsTypes
__all__ = ["FlextTestsTypes", "t"]
