"""FlextUtilities - Pure Facade for FLEXT Utility Classes.

This module provides a unified entry point to all FLEXT utility functionality.
All methods are delegated to specialized classes in _utilities/ submodules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence, Sized
from enum import StrEnum
from typing import TypeIs, overload

# NOTE: cast() removed - use type narrowing and Protocols instead
from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.cast import FlextUtilitiesCast, cast_safe as _cast_safe_fn
from flext_core._utilities.checker import FlextUtilitiesChecker
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.configuration import FlextUtilitiesConfiguration
from flext_core._utilities.context import FlextUtilitiesContext
from flext_core._utilities.conversion import (
    FlextUtilitiesConversion,
    conversion as _conversion_fn,
)
from flext_core._utilities.deprecation import FlextUtilitiesDeprecation
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core._utilities.enum import FlextUtilitiesEnum, enum as _enum_fn
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core._utilities.pagination import FlextUtilitiesPagination
from flext_core._utilities.parser import FlextUtilitiesParser
from flext_core._utilities.reliability import FlextUtilitiesReliability
from flext_core._utilities.text import FlextUtilitiesText
from flext_core._utilities.validation import FlextUtilitiesValidation
from flext_core._utilities.validators import ValidatorDSL, ValidatorSpec
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.typings import t


class FlextUtilities:
    """Unified facade for all FLEXT utility functionality.

    This class provides access to specialized utility classes through
    class attributes. All functionality is delegated to the appropriate
    specialized class.

    Usage:
        from flext_core import FlextUtilities

        # Access via facade attributes
        result = FlextUtilities.Parser.parse(value, int)
        result = FlextUtilities.Collection.map(items, fn)
        result = FlextUtilities.Mapper.get(data, "key")

        # Or use short alias
        from flext_core.utilities import u
        result = u.Parser.parse(value, int)
    """

    # === FACADE ATTRIBUTES - Real inheritance classes ===
    class Args(FlextUtilitiesArgs):
        """Args utility class - real inheritance."""

    class Cache(FlextUtilitiesCache):
        """Cache utility class - real inheritance."""

    class Checker(FlextUtilitiesChecker):
        """Checker utility class - real inheritance."""

    class Collection(FlextUtilitiesCollection):
        """Collection utility class - real inheritance."""

    class Configuration(FlextUtilitiesConfiguration):
        """Configuration utility class - real inheritance."""

    class Cast(FlextUtilitiesCast):
        """Cast utility class - real inheritance."""

    class Context(FlextUtilitiesContext):
        """Context utility class - real inheritance."""

    class Conversion(FlextUtilitiesConversion):
        """Conversion utility class - real inheritance."""

    class Deprecation(FlextUtilitiesDeprecation):
        """Deprecation utility class - real inheritance."""

    class Domain(FlextUtilitiesDomain):
        """Domain utility class - real inheritance."""

    class Enum(FlextUtilitiesEnum):
        """Enum utility class - real inheritance."""

    # Generalized public function aliases
    enum = staticmethod(_enum_fn)
    conversion = staticmethod(_conversion_fn)
    cast_safe = staticmethod(_cast_safe_fn)

    # Convenience shortcuts for common operations
    @staticmethod
    def is_enum_member[E: StrEnum](value: object, enum_cls: type[E]) -> TypeIs[E]:
        """Check if value is enum member. Alias for enum(value, cls, mode='is_member')."""
        return _enum_fn(value, enum_cls, mode="is_member")

    @staticmethod
    def parse_enum[E: StrEnum](enum_cls: type[E], value: str | E) -> r[E]:
        """Parse value to enum. Alias for enum(value, cls, mode='parse')."""
        return _enum_fn(value, enum_cls, mode="parse")

    @staticmethod
    def to_str(value: object, *, default: str | None = None) -> str:
        """Convert value to string. Alias for conversion(value, mode='to_str')."""
        return _conversion_fn(value, mode="to_str", default=default)

    @staticmethod
    def to_str_list(
        value: object,
        *,
        default: list[str] | None = None,
    ) -> list[str]:
        """Convert value to list of strings. Alias for conversion(value, mode='to_str_list')."""
        return _conversion_fn(value, mode="to_str_list", default=default)

    @staticmethod
    def mapper() -> FlextUtilitiesMapper:
        """Get FlextUtilitiesMapper instance.

        Preferred access method for mapper utilities.
        Use u.mapper() instead of u.Mapper for better encapsulation.

        Returns:
            FlextUtilitiesMapper instance

        Example:
            >>> from flext_core.utilities import u
            >>> mapper = u.mapper()
            >>> result = mapper.get(data, "key", default="")
            >>> extracted = mapper.extract(config, "database.port")

        """
        return FlextUtilitiesMapper()

    class Generators(FlextUtilitiesGenerators):
        """Generators utility class - real inheritance."""

    class Guards(FlextUtilitiesGuards):
        """Guards utility class - real inheritance.

        Provides public access to type guard methods. All methods delegate
        to private methods in FlextUtilitiesGuards to enforce proper API usage.
        """

        @staticmethod
        def normalize_to_metadata_value(
            val: t.GeneralValueType,
        ) -> t.MetadataAttributeValue:
            """Normalize any value to MetadataAttributeValue.

            Public wrapper for normalize_to_metadata_value().

            Args:
                val: Value to normalize

            Returns:
                t.MetadataAttributeValue: Normalized value compatible with Metadata attributes

            Example:
                >>> from flext_core.utilities import u
                >>> u.Guards.normalize_to_metadata_value("test")
                'test'

            """
            return FlextUtilitiesGuards.normalize_to_metadata_value(val)

        @staticmethod
        def is_string_non_empty(value: t.GeneralValueType) -> bool:
            """Check if value is a non-empty string.

            Public wrapper for is_string_non_empty().

            Args:
                value: Object to check

            Returns:
                bool: True if value is non-empty string

            """
            return FlextUtilitiesGuards.is_string_non_empty(value)

        @staticmethod
        def is_dict_non_empty(value: t.GeneralValueType) -> bool:
            """Check if value is a non-empty dictionary.

            Public wrapper for is_dict_non_empty().

            Args:
                value: Object to check

            Returns:
                bool: True if value is non-empty dict-like

            """
            return FlextUtilitiesGuards.is_dict_non_empty(value)

        @staticmethod
        def is_list_non_empty(value: t.GeneralValueType) -> bool:
            """Check if value is a non-empty list.

            Public wrapper for is_list_non_empty().

            Args:
                value: Object to check

            Returns:
                bool: True if value is non-empty list-like

            """
            return FlextUtilitiesGuards.is_list_non_empty(value)

    class Mapper(FlextUtilitiesMapper):
        """Mapper utility class - real inheritance."""

    class Model(FlextUtilitiesModel):
        """Model utility class - real inheritance."""

    class Pagination(FlextUtilitiesPagination):
        """Pagination utility class - real inheritance."""

    class Parser(FlextUtilitiesParser):
        """Parser utility class - real inheritance."""

    class Reliability(FlextUtilitiesReliability):
        """Reliability utility class - real inheritance."""

    class Text(FlextUtilitiesText):
        """Text utility class - real inheritance."""

    class Validation(FlextUtilitiesValidation):
        """Validation utility class - real inheritance."""

    class Validators(ValidatorDSL):
        """Validators utility class - real inheritance."""

    class V(ValidatorDSL):
        """V utility class - alias for Validators - real inheritance."""

    class Pipeline:
        """Pipeline utility for chaining operations with FlextResult.

        Provides convenient access to FlextResult pipeline methods for
        composing operations in a functional style.
        """

        @staticmethod
        def flow_through[T, U](
            result: r[T],
            *funcs: Callable[[T | U], r[U]],
        ) -> r[U]:
            """Chain multiple operations in a pipeline.

            Args:
                result: Initial FlextResult to start the pipeline
                *funcs: Functions to apply in sequence

            Returns:
                Final FlextResult after all transformations

            Example:
                >>> result = r[int].ok(5)
                >>> final = Pipeline.flow_through(
                ...     result, lambda x: r[int].ok(x * 2), lambda x: r[str].ok(str(x))
                ... )

            """
            return result.flow_through(*funcs)

        @staticmethod
        def and_then[T, U](
            result: r[T],
            func: Callable[[T], r[U]],
        ) -> r[U]:
            """Chain a single operation.

            Args:
                result: Initial FlextResult
                func: Function to apply

            Returns:
                Result after transformation

            """
            return result.and_then(func)

        @staticmethod
        def fold[T, U](
            result: r[T],
            on_failure: Callable[[str], U],
            on_success: Callable[[T], U],
        ) -> U:
            """Fold result into a single value.

            Args:
                result: FlextResult to fold
                on_failure: Function to call on failure
                on_success: Function to call on success

            Returns:
                Folded value

            """
            return result.fold(on_failure, on_success)

        @staticmethod
        def tap[T](
            result: r[T],
            func: Callable[[T], None],
        ) -> r[T]:
            """Execute side effect without changing result.

            Args:
                result: FlextResult
                func: Side effect function

            Returns:
                Same result (for chaining)

            """
            return result.tap(func)

    class Cloning:
        """Cloning utility for runtime and container instances.

        Provides convenient access to cloning operations using protocols
        to avoid circular imports.
        """

        @staticmethod
        def clone_runtime[T](
            runtime: T,
            *,
            context: p.Ctx | None = None,
            config_overrides: dict[str, object] | None = None,
        ) -> T:
            """Clone runtime with optional overrides.

            Creates a new runtime instance with the same dispatcher and registry,
            but with optional context and config overrides.

            Args:
                runtime: Runtime instance to clone (must implement Runtime protocol).
                context: Optional new context. If not provided, uses runtime's context.
                config_overrides: Optional config field overrides.

            Returns:
                T: Cloned runtime instance.

            Example:
                >>> new_runtime = FlextUtilities.Cloning.clone_runtime(
                ...     existing_runtime,
                ...     context=new_context,
                ...     config_overrides={"log_level": "DEBUG"},
                ... )

            """
            # Create new instance without calling __init__
            cloned: T = runtime.__class__.__new__(runtime.__class__)
            # Copy dispatcher and registry via protocol attributes
            # Note: Accessing private attributes is necessary for cloning runtime instances
            # that implement the Runtime protocol. These attributes are part of the
            # internal implementation and are accessed during cloning operations.
            # Use setattr with variables to avoid type checker errors for private attributes
            if hasattr(runtime, "_dispatcher"):
                dispatcher_attr = "_dispatcher"
                setattr(cloned, dispatcher_attr, getattr(runtime, dispatcher_attr))
            if hasattr(runtime, "_registry"):
                registry_attr = "_registry"
                setattr(cloned, registry_attr, getattr(runtime, registry_attr))
            # Use new context or copy existing
            if hasattr(runtime, "_context"):
                context_attr = "_context"
                cloned_context = context or getattr(runtime, context_attr)
                setattr(cloned, context_attr, cloned_context)
            # Clone config with overrides
            if hasattr(runtime, "_config"):
                config_attr = "_config"
                runtime_config = getattr(runtime, config_attr)
                if config_overrides:
                    cloned_config = runtime_config.model_copy(update=config_overrides)
                    setattr(cloned, config_attr, cloned_config)
                else:
                    setattr(cloned, config_attr, runtime_config)
            return cloned

        @staticmethod
        def clone_container(
            container: p.DI,
            *,
            scope_id: str | None = None,
            overrides: dict[str, object] | None = None,
        ) -> p.DI:
            """Clone container with scoping.

            Creates a scoped container instance with optional service overrides.

            Args:
                container: Container instance to clone (must implement DI protocol).
                scope_id: Optional scope identifier.
                overrides: Optional service overrides.

            Returns:
                p.DI: Scoped container instance.

            Example:
                >>> scoped = FlextUtilities.Cloning.clone_container(
                ...     global_container,
                ...     scope_id="worker_1",
                ...     overrides={"logger": custom_logger},
                ... )

            """
            # Use container's scoped() method for proper scoping
            # Python 3.13: dict[str, object] is structurally compatible with ServiceMapping
            # ServiceInstanceType = GeneralValueType | BaseModel | Callable[..., GeneralValueType] | object
            # object is compatible with ServiceInstanceType (object is included in union)
            # Convert dict to Mapping for structural compatibility
            services_mapping: t.Types.ServiceMapping | None = (
                dict(overrides) if overrides is not None else None
            )
            return container.scoped(
                subproject=scope_id,
                services=services_mapping,
            )

    class Registration:
        """Registration utility for container services.

        Provides convenient access to registration operations using protocols
        to avoid circular imports.
        """

        @staticmethod
        def register_singleton[T](
            container: p.DI,
            name: str,
            instance: T,
        ) -> r[None]:
            """Register singleton with standard error handling.

            Args:
                container: Container to register in (must implement DI protocol).
                name: Service name.
                instance: Service instance to register.

            Returns:
                r[None]: Success if registration succeeds, failure otherwise.

            Example:
                >>> result = FlextUtilities.Registration.register_singleton(
                ...     container, "db", DatabaseService()
                ... )

            """
            try:
                # Python 3.13: T is compatible with ServiceInstanceType via structural typing
                # ServiceInstanceType = GeneralValueType | BaseModel | Callable[..., GeneralValueType] | object
                # T (generic) is compatible if it matches any of these types
                # Use Protocol check for type narrowing - if T has required structure, it's compatible
                # Direct assignment works - type checker recognizes structural compatibility
                register_result = container.register(name, instance)
                if register_result.is_failure:
                    return r[None].fail(
                        register_result.error or "Registration failed",
                    )
                # For None values, we need to create directly since ok() doesn't accept None
                return r[None].ok(None)
            except Exception as e:
                return r[None].fail(f"Registration failed for {name}: {e}")

        @staticmethod
        def register_factory[T](
            container: p.DI,
            name: str,
            factory: Callable[[], T],
            *,
            _cache: bool = False,
        ) -> r[None]:
            """Register factory with optional caching.

            Args:
                container: Container to register in (must implement DI protocol).
                name: Factory name.
                factory: Factory function to register.
                _cache: Reserved for future implementation of cached factory pattern.

            Returns:
                r[None]: Success if registration succeeds, failure otherwise.

            Note:
                The _cache parameter is reserved for future implementation of
                cached factory pattern.

            Example:
                >>> result = FlextUtilities.Registration.register_factory(
                ...     container, "logger", create_logger, _cache=True
                ... )

            """
            try:
                # Python 3.13: Callable[[], T] is compatible with FactoryCallable = Callable[[], object]
                # T is compatible with object (all types are compatible with object)
                # Direct assignment works - type checker recognizes callable compatibility
                # FactoryCallable accepts any zero-arg callable returning any object
                register_result = container.register_factory(name, factory)
                if register_result.is_failure:
                    return r[None].fail(
                        register_result.error or "Factory registration failed",
                    )
                # For None values, we need to create directly since ok() doesn't accept None
                return r[None].ok(None)
            except Exception as e:
                return r[None].fail(
                    f"Factory registration failed for {name}: {e}",
                )

        @staticmethod
        def bulk_register(
            container: p.DI,
            registrations: Mapping[
                str,
                object | Callable[[], t.GeneralValueType],
            ],
        ) -> r[int]:
            """Register multiple services at once.

            Args:
                container: Container to register in (must implement DI protocol).
                registrations: Mapping of name to service instance or factory.

            Returns:
                r[int]: Success with count of registered services, or failure.

            Example:
                >>> result = FlextUtilities.Registration.bulk_register(
                ...     container,
                ...     {
                ...         "db": DatabaseService(),
                ...         "logger": create_logger,
                ...     },
                ... )

            """
            count = 0
            for name, value in registrations.items():
                try:
                    if callable(value):
                        # Python 3.13: callable value is compatible with FactoryCallable = Callable[[], object]
                        # Direct assignment works - type checker recognizes callable compatibility
                        register_result = container.register_factory(name, value)
                    else:
                        # Python 3.13: object is compatible with ServiceInstanceType
                        # ServiceInstanceType = GeneralValueType | BaseModel | Callable[..., GeneralValueType] | object
                        # object is included in union, so direct assignment works
                        register_result = container.register(name, value)
                    if register_result.is_failure:
                        return r[int].fail(
                            f"Bulk registration failed at {name}: {register_result.error}",
                        )
                    count += 1
                except Exception as e:
                    return r[int].fail(
                        f"Bulk registration failed at {name}: {e}",
                    )
            return r[int].ok(count)

    # === UTILITY METHODS ===
    # These are convenience methods that delegate to specialized classes.
    # Access specialized classes directly: FlextUtilities.Parser.parse(), etc.

    @staticmethod
    def is_type(value: object, type_spec: str | type | tuple[type, ...]) -> bool:
        """Generic type checking function that unifies all guard checks.

        Provides a single entry point for all type checking operations,
        supporting string-based type names, direct type/class checks, and
        protocol checks. This function delegates to FlextUtilitiesGuards.is_type().

        Args:
            value: Object to check
            type_spec: Type specification as:
                - String name: "config", "str", "dict", "list", "sequence",
                  "mapping", "callable", "sized", "list_or_tuple", "sequence_not_str",
                  "string_non_empty", "dict_non_empty", "list_non_empty"
                - Type/class: str, dict, list, tuple, Sequence, Mapping, etc.
                - Tuple of types: (int, float), (str, bytes), etc.
                - Protocol: p.Config, p.Ctx, etc.

        Returns:
            bool: True if value matches the type specification

        Examples:
            >>> # String-based checks
            >>> u.is_type(obj, "config")
            >>> u.is_type(obj, "str")
            >>> u.is_type(obj, "dict")
            >>> u.is_type(obj, "string_non_empty")

            >>> # Direct type checks
            >>> u.is_type(obj, str)
            >>> u.is_type(obj, dict)
            >>> u.is_type(obj, list)

            >>> # Tuple of types checks
            >>> u.is_type(obj, (int, float))
            >>> u.is_type(obj, (str, bytes))

            >>> # Protocol checks
            >>> u.is_type(obj, p.Config)
            >>> u.is_type(obj, p.Ctx)

        """
        return FlextUtilitiesGuards.is_type(value, type_spec)

    @staticmethod
    def merge(
        base: t.Types.ConfigurationDict,
        other: t.Types.ConfigurationDict,
        *,
        strategy: str = "deep",
    ) -> r[t.Types.ConfigurationDict]:
        """Merge two dictionaries - delegates to FlextUtilitiesCollection."""
        return FlextUtilitiesCollection.merge(base, other, strategy=strategy)

    @staticmethod
    def transform(
        source: t.Types.ConfigurationDict | t.Types.ConfigurationMapping,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: t.Types.StringDict | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
        to_json: bool = False,
    ) -> r[t.Types.ConfigurationDict]:
        """Transform dictionary with multiple options - delegates to FlextUtilitiesMapper.

        Args:
            source: Source dictionary to transform.
            normalize: Normalize values using cache normalization.
            strip_none: Remove keys with None values.
            strip_empty: Remove keys with empty values (empty strings, lists, dicts).
            map_keys: Dictionary mapping old keys to new keys.
            filter_keys: Set of keys to keep (all others removed).
            exclude_keys: Set of keys to remove.
            to_json: Convert to JSON-compatible values.

        Returns:
            FlextResult with transformed dictionary.

        """
        return FlextUtilitiesMapper.transform(
            source,
            normalize=normalize,
            strip_none=strip_none,
            strip_empty=strip_empty,
            map_keys=map_keys,
            filter_keys=filter_keys,
            exclude_keys=exclude_keys,
            to_json=to_json,
        )

    # Result helpers - FlextResult DSL methods
    @staticmethod
    def val[T](result: p.Result[T], default: T) -> T:
        """Extract value from FlextResult or RuntimeResult with default fallback.

        Accepts both FlextResult[T] and RuntimeResult[T] from flext-core bootstrap layer.
        Both implement p.Result[T] protocol interface.
        """
        # Use .value directly - FlextResult/RuntimeResult never return None on success
        return result.value if result.is_success else default

    @staticmethod
    def result_val[T](result: p.Result[T], default: T) -> T:
        """Extract value from FlextResult or RuntimeResult with default fallback (alias for val)."""
        return u.val(result, default)

    @staticmethod
    def vals[T](results: Sequence[p.Result[T]]) -> list[T]:
        """Extract values from collection of FlextResult or RuntimeResult objects, skipping failures."""
        # Use .value directly - FlextResult/RuntimeResult never return None on success
        return [r.value for r in results if r.is_success]

    @staticmethod
    def err[T](result: p.Result[T], default: str = "") -> str:
        """Get error message from FlextResult or RuntimeResult.

        Accepts both FlextResult[T] and RuntimeResult[T] from flext-core bootstrap layer.
        When is_failure is True, error is never None (fail() converts None to "").
        """
        if result.is_failure:
            return result.error or default
        return default

    @staticmethod
    def generate(
        kind: str | None = None,
        *,
        prefix: str | None = None,
        parts: tuple[t.GeneralValueType, ...] | None = None,
        length: int | None = None,
        include_timestamp: bool = False,
        separator: str = "_",
    ) -> str:
        """Generate ID by kind or custom prefix - delegates to FlextUtilitiesGenerators.

        Args:
            kind: ID kind ("uuid", "correlation", "entity", "batch", "transaction",
                "saga", "event", "command", "query", "aggregate", "ulid", "id").
                If None, generates UUID.
            prefix: Custom prefix (overrides kind prefix if provided).
            parts: Additional parts to include in ID (e.g., batch_size, aggregate_type).
            length: Custom length for generated ID (only for ulid/short IDs).
            include_timestamp: Include timestamp in ID.
            separator: Separator between prefix, parts, and ID (default: "_").

        Returns:
            Generated ID string.

        Examples:
            >>> u.generate()  # UUID (36 chars)
            >>> u.generate("uuid")  # UUID (36 chars)
            >>> u.generate("correlation")  # corr_...
            >>> u.generate("entity", prefix="user")  # user_...
            >>> u.generate("batch", parts=(100,))  # batch_100_...
            >>> u.generate("aggregate", prefix="user")  # user_...
            >>> u.generate("ulid", length=16)  # Short ID with 16 chars

        """
        # Delegate to generators.py - all logic is there
        return FlextUtilitiesGenerators.generate(
            kind=kind,
            prefix=prefix,
            parts=parts,
            length=length,
            include_timestamp=include_timestamp,
            separator=separator,
        )

    @staticmethod
    def extract[T](
        data: t.Types.ConfigurationMapping | object,
        path: str,
        *,
        default: T | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> r[T | None]:
        """Extract nested value from data structure - delegates to FlextUtilitiesMapper.

        Args:
            data: Source data (dict, object with attributes, or Pydantic model).
            path: Dot-separated path (e.g., "user.profile.name").
            default: Default value if path not found.
            required: Fail if path not found.
            separator: Path separator (default: ".").

        Returns:
            FlextResult containing extracted value or default.

        Example:
            >>> result = u.extract({"user": {"name": "John"}}, "user.name")
            >>> value = result.value  # "John"

        """
        return FlextUtilitiesMapper.extract(
            data,
            path,
            default=default,
            required=required,
            separator=separator,
        )

    @staticmethod
    def has(obj: object, key: str) -> bool:
        """Check if object has attribute/key."""
        if isinstance(obj, dict):
            return key in obj
        return hasattr(obj, key)

    # =========================================================================
    # UNIVERSAL CHECK METHODS - Highly generalized and parametrizable
    # =========================================================================

    @staticmethod
    def chk(
        value: object,
        *,
        eq: object | None = None,
        ne: object | None = None,
        gt: float | None = None,
        gte: float | None = None,
        lt: float | None = None,
        lte: float | None = None,
        is_: type[object] | None = None,
        not_: type[object] | None = None,
        in_: Sequence[object] | None = None,
        not_in: Sequence[object] | None = None,
        none: bool | None = None,
        empty: bool | None = None,
        match: str | None = None,
        contains: str | object | None = None,
        starts: str | None = None,
        ends: str | None = None,
    ) -> bool:
        """Universal check - single method for ALL validation scenarios.

        Args:
            value: Value to check
            eq: Check value == eq
            ne: Check value != ne
            gt/gte/lt/lte: Numeric comparisons (works with len for sequences)
            is_: Check isinstance(value, is_)
            not_: Check not isinstance(value, not_)
            in_: Check value in in_
            not_in: Check value not in not_in
            none: Check value is None (True) or is not None (False)
            empty: Check if empty (True) or not empty (False)
            match: Check regex pattern match (strings)
            contains: Check if value contains item
            starts/ends: Check string prefix/suffix

        Returns:
            True if ALL conditions pass, False otherwise.

        Examples:
            u.chk(x, gt=0, lt=100)             # 0 < x < 100
            u.chk(s, empty=False, match="[0-9]+")  # non-empty and has digits
            u.chk(lst, gte=1, lte=10)          # 1 <= len(lst) <= 10
            u.chk(v, is_=str, none=False)      # is string and not None

        """
        # None checks
        if none is True and value is not None:
            return False
        if none is False and value is None:
            return False

        # Type checks
        if is_ is not None and not isinstance(value, is_):
            return False
        if not_ is not None and isinstance(value, not_):
            return False

        # Equality checks
        if eq is not None and value != eq:
            return False
        if ne is not None and value == ne:
            return False

        # Membership checks
        if in_ is not None and value not in in_:
            return False
        if not_in is not None and value in not_in:
            return False

        # Length/numeric checks - use len() for sequences, direct for numbers
        # Python 3.13: Use isinstance for proper type narrowing
        check_val: int | float
        if isinstance(value, (int, float)):
            check_val = value
        elif isinstance(value, (Sequence, Mapping)):
            # Type narrowing: Sequence and Mapping have __len__
            check_val = len(value)
        elif value is not None and isinstance(value, Sized):
            # Type narrowing: value implements Sized protocol (has __len__)
            # Sized protocol ensures __len__ exists and returns int
            check_val = len(value)
        else:
            check_val = 0

        if gt is not None and check_val <= gt:
            return False
        if gte is not None and check_val < gte:
            return False
        if lt is not None and check_val >= lt:
            return False
        if lte is not None and check_val > lte:
            return False

        # Empty checks (after len is computed)
        if empty is True and check_val != 0:
            return False
        if empty is False and check_val == 0:
            return False

        # String-specific checks
        if isinstance(value, str):
            if match is not None and not re.search(match, value):
                return False
            if starts is not None and not value.startswith(starts):
                return False
            if ends is not None and not value.endswith(ends):
                return False
            if (
                contains is not None
                and isinstance(contains, str)
                and contains not in value
            ):
                return False
        elif contains is not None:
            # Generic containment for sequences/dicts
            if isinstance(value, dict):
                # Type narrowing: value is dict after isinstance check
                # dict supports 'in' operator for keys - no cast needed
                if contains not in value:
                    return False
            elif value is not None and hasattr(value, "__contains__"):
                # Type narrowing: value is not None and has __contains__ method
                # Check containment using getattr for type safety
                contains_method = getattr(value, "__contains__", None)
                if contains_method is not None:
                    try:
                        if not contains_method(contains):
                            return False
                    except (TypeError, ValueError):
                        # If containment check fails due to type mismatch, consider it not contained
                        return False

        return True

    @staticmethod
    def normalize(text: str, pattern: str = r"\s+", replacement: str = " ") -> str:
        """Normalize whitespace - delegates to FlextUtilitiesParser."""
        parser = FlextUtilitiesParser()
        result = parser.normalize_whitespace(
            text,
            pattern=pattern,
            replacement=replacement,
        )
        # Use .value directly - FlextResult never returns None on success
        normalized = result.value if result.is_success else text
        return normalized if u.is_type(normalized, str) else text

    # Power methods - convenience delegates
    @staticmethod
    def pipe(
        value: object,
        *operations: Callable[[object], object],
        on_error: str = "stop",
    ) -> r[object]:
        """Functional pipeline - delegates to FlextUtilitiesReliability.pipe."""
        return FlextUtilitiesReliability.pipe(value, *operations, on_error=on_error)

    @staticmethod
    def batch[T, R](
        items: list[T],
        operation: Callable[[T], R | r[R]],
        *,
        _size: int = c.DEFAULT_BATCH_SIZE,
        on_error: str = "collect",
        _parallel: bool = False,
        progress: Callable[[int, int], None] | None = None,
        _progress_interval: int = 1,
        pre_validate: Callable[[T], bool] | None = None,
        flatten: bool = False,
    ) -> r[t.Types.BatchResultDict]:
        """Process items in batches - delegates to FlextUtilitiesCollection.batch."""
        return FlextUtilitiesCollection.batch(
            items,
            operation,
            _size=_size,
            on_error=on_error,
            _parallel=_parallel,
            progress=progress,
            _progress_interval=_progress_interval,
            pre_validate=pre_validate,
            flatten=flatten,
        )

    @staticmethod
    def retry[TResult](
        operation: Callable[[], r[TResult] | TResult],
        max_attempts: int | None = None,
        delay: float | None = None,
        delay_seconds: float | None = None,
        backoff_multiplier: float | None = None,
        retry_on: tuple[type[Exception], ...] | None = None,
    ) -> r[TResult]:
        """Execute operation with retry logic - delegates to FlextUtilitiesReliability.retry."""
        return FlextUtilitiesReliability.retry(
            operation,
            max_attempts=max_attempts,
            delay=delay,
            delay_seconds=delay_seconds,
            backoff_multiplier=backoff_multiplier,
            retry_on=retry_on,
        )

    @staticmethod
    @overload
    def get(
        data: t.Types.ConfigurationMapping | object,
        key: str,
        *,
        default: str = "",
    ) -> str: ...

    @staticmethod
    @overload
    def get[T](
        data: t.Types.ConfigurationMapping | object,
        key: str,
        *,
        default: list[T],
    ) -> list[T]: ...

    @staticmethod
    @overload
    def get[T](
        data: t.Types.ConfigurationMapping | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None: ...

    @staticmethod
    def get[T](
        data: t.Types.ConfigurationMapping | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None:
        """Get value from dict/object - delegates to FlextUtilitiesMapper.get."""
        return FlextUtilitiesMapper.get(data, key, default=default)


u = FlextUtilities  # Runtime alias (not TypeAlias to avoid PYI042)
u_core = FlextUtilities  # Runtime alias (not TypeAlias to avoid PYI042)

__all__ = [
    "FlextUtilities",
    "ValidatorSpec",  # Export for flext-ldif and other projects
    "u",
    "u_core",
]
