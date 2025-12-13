"""Test data builders for FLEXT ecosystem tests.

Provides ultra-powerful builder pattern for creating complex test data structures.
Supports FlextResult, lists, dicts, mappings, and generic classes with fluent interface.

Uses flext-core utilities extensively to avoid code duplication.
Designed with minimal public methods that handle almost everything.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Self, TypeGuard, cast, overload

from pydantic import BaseModel

from flext_core import (
    FlextResult as r,
    p,
    u,
)
from flext_core.runtime import FlextRuntime
from flext_core.typings import T
from flext_tests.constants import c
from flext_tests.factories import FlextTestsFactories as tt
from flext_tests.models import FlextTestsModels as m
from flext_tests.typings import t
from flext_tests.utilities import FlextTestsUtilities as tu


class FlextTestsBuilders:
    """Ultra-powerful test data builder with fluent interface.

    Provides minimal public methods that handle almost everything:
    - add(): Universal method to add any data type with smart inference
    - set(): Set value at nested path
    - get(): Get value from path
    - to_result(): Build as FlextResult
    - build(): Build with output type control

    Supports:
    - FlextResult wrapping
    - Lists and sequences
    - Dicts and mappings
    - Pydantic models
    - Generic classes
    - Transformations and validations
    - Nested structures

    Example:
        from flexts import tb

        # Simple usage
        dataset = tb().add("users", count=5).add("config", production=True).build()

        # With FlextResult
        result = tb().add("data", value=42).to_result()

        # With model output
        model = tb().add("name", "test").add("value", 100).build(as_model=MyModel)

        # Complex nested structures
        dataset = (
            tb()
            .add("users", factory="users", count=3)
            .add("settings", mapping={"debug": True, "timeout": 30})
            .set("settings.advanced.feature", enabled=True)
            .build()
        )

    """

    # Use class attribute (not PrivateAttr) to match FlextService pattern
    # Initialize as None to avoid ClassVar requirement (mutable default)
    _data: t.Tests.Builders.BuilderDict | None = None

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize builder with optional initial data."""
        # Initialize without inheritance
        # Set attribute directly (no PrivateAttr needed, compatible with FlextService)
        # Always initialize as empty dict (class attribute is default, instance needs fresh copy)
        self._data = {}

    def _ensure_data_initialized(self) -> None:
        """Ensure _data is initialized (helper for type safety)."""
        if self._data is None:
            self._data = {}
        # Type assertion for mypy - after this method, _data is guaranteed to be dict
        assert self._data is not None, "_data must be initialized"

    def execute(self) -> r[t.Tests.Builders.BuilderDict]:
        """Execute service - builds and returns as FlextResult.

        Returns:
            FlextResult containing built data.

        """
        self._ensure_data_initialized()
        data = self.build()
        # Return as BuilderDict
        return r[t.Tests.Builders.BuilderDict].ok(data)

    # =========================================================================
    # CORE PUBLIC METHODS - Minimal and Powerful
    # =========================================================================

    def add(
        self,
        key: str,
        value: t.Tests.Builders.BuilderValue | None = None,
        **kwargs: object,  # Accept object to allow Callable, set, etc. - validated by AddParams
    ) -> Self:
        """Add data to builder with smart type inference.

        Uses Pydantic 2 models for parameter validation and computation.
        All parameters are validated using m.Tests.Builders.AddParams model.

        Resolution order (first match wins):
        1. result → Store FlextResult as-is
        2. result_ok → r[T].ok(result_ok)
        3. result_fail → r[T].fail(result_fail, error_code=result_code)
        4. results → Store list of FlextResult
        5. results_ok → [r[T].ok(v) for v in results_ok]
        6. results_fail → [r[T].fail(e) for e in results_fail]
        7. cls → cls(*cls_args, **cls_kwargs)
        8. items → Apply items_map/items_filter, store list
        9. entries → Apply entries_map/entries_filter, store dict
        10. factory → Use FlextTestsFactories (existing)
        11. model → Pydantic model instantiation (existing)
        12. mapping → Store as dict (existing)
        13. sequence → Store as list (existing)
        14. value → Direct value (existing)
        15. default → Fallback (existing)

        Args:
            key: Key to store data under.
            value: Direct value to store.
            **kwargs: Additional parameters (factory, count, model, etc.)

        Returns:
            Self for method chaining.

        Examples:
            # Direct value
            tb().add("name", "test")

            # With FlextResult
            tb().add("result", result_ok=42)
            tb().add("error", result_fail="Failed", result_code="E001")

            # With generic class
            tb().add("instance", cls=MyClass, cls_kwargs={"x": 1})

            # With items transformation
            tb().add("doubled", items=[1, 2, 3], items_map=lambda x: x * 2)

        """
        # Convert kwargs to validated model using FlextUtilities
        # u.Model.from_kwargs() accepts **kwargs: object - Pydantic 2 handles conversions automatically
        params_result = u.Model.from_kwargs(
            m.Tests.Builders.AddParams,
            key=key,
            value=value,
            **kwargs,
        )
        if params_result.is_failure:
            error_msg = f"Invalid add() parameters: {params_result.error}"
            raise ValueError(error_msg)
        params = params_result.value

        resolved_value: t.Tests.Builders.BuilderValue = None

        # Priority 1: FlextResult direct
        if params.result is not None:
            resolved_value = cast("t.Tests.Builders.BuilderValue", params.result)

        # Priority 2: Create success result
        elif params.result_ok is not None:
            error_code_val = params.result_code or c.Errors.VALIDATION_ERROR
            resolved_value = cast(
                "t.Tests.Builders.BuilderValue",
                FlextRuntime.RuntimeResult[t.GeneralValueType].ok(params.result_ok),
            )

        # Priority 3: Create failure result
        elif params.result_fail is not None:
            error_code_val = params.result_code or c.Errors.VALIDATION_ERROR
            resolved_value = cast(
                "t.Tests.Builders.BuilderValue",
                FlextRuntime.RuntimeResult[t.GeneralValueType].fail(
                    params.result_fail,
                    error_code=error_code_val,
                ),
            )

        # Priority 4: Batch results
        elif params.results is not None:
            resolved_value = cast(
                "t.Tests.Builders.BuilderValue",
                list(params.results),
            )

        # Priority 5: Batch success results
        elif params.results_ok is not None:
            resolved_value = cast(
                "t.Tests.Builders.BuilderValue",
                [r[t.GeneralValueType].ok(v) for v in params.results_ok],
            )

        # Priority 6: Batch failure results
        elif params.results_fail is not None:
            error_code_val = params.result_code or c.Errors.VALIDATION_ERROR
            resolved_value = cast(
                "t.Tests.Builders.BuilderValue",
                [
                    r[t.GeneralValueType].fail(e, error_code=error_code_val)
                    for e in params.results_fail
                ],
            )

        # Priority 7: Generic class instantiation
        elif params.cls is not None:
            args = params.cls_args or ()
            cls_kwargs = params.cls_kwargs or {}
            # Check if it's a domain entity or value object
            cls_type = params.cls

            # Type guards for Entity and Value classes
            def is_entity_class(
                cls: type[object],
            ) -> TypeGuard[type[m.Entity]]:
                """Type guard to check if class is Entity subclass."""
                return issubclass(cls, m.Entity)

            def is_value_class(
                cls: type[object],
            ) -> TypeGuard[type[m.Value]]:
                """Type guard to check if class is Value subclass."""
                return issubclass(cls, m.Value)

            # Type narrowing for Entity classes
            if is_entity_class(cls_type):
                name_val = cls_kwargs.get("name", "")
                value_val = cls_kwargs.get("value", "")
                resolved_value = tu.Tests.DomainHelpers.create_test_entity_instance(
                    name=str(name_val) if name_val else "",
                    value=value_val or "",
                    entity_class=cls_type,
                )
            # Type narrowing for Value classes
            elif is_value_class(cls_type):
                data_val = cls_kwargs.get("data", "")
                count_val = cls_kwargs.get("count", 1)
                resolved_value = (
                    tu.Tests.DomainHelpers.create_test_value_object_instance(
                        data=str(data_val) if data_val else "",
                        count=int(count_val)
                        if isinstance(count_val, (int, float))
                        else 1,
                        value_class=cls_type,
                    )
                )
            # Generic class instantiation
            # Type checker cannot infer exact class type from cls_type
            # We use direct instantiation with runtime validation
            else:
                # Direct instantiation - type checker limitation with dynamic types
                # This is safe because params.cls is validated to be a type
                # Runtime will work correctly even if type checker can't verify
                if args or cls_kwargs:
                    # Use getattr to help type checker understand it's callable
                    instance = cls_type.__call__(*args, **cls_kwargs)
                else:
                    instance = cls_type.__call__()
                resolved_value = cast("t.Tests.Builders.BuilderValue", instance)

        # Priority 8: Items with transformation/filtering
        elif params.items is not None:
            items_processed = list(params.items)
            if params.items_filter is not None:
                items_processed = [
                    item for item in items_processed if params.items_filter(item)
                ]
            if params.items_map is not None:
                items_processed = [params.items_map(item) for item in items_processed]
            resolved_value = cast("t.Tests.Builders.BuilderValue", items_processed)

        # Priority 9: Entries with transformation/filtering
        elif params.entries is not None:
            entries_processed: dict[str, t.GeneralValueType] = dict(params.entries)
            if params.entries_filter is not None:
                entries_processed = {
                    k: v
                    for k, v in entries_processed.items()
                    if k in params.entries_filter
                }
            if params.entries_map is not None:
                entries_processed = {
                    k: params.entries_map(v) for k, v in entries_processed.items()
                }
            resolved_value = cast(
                "t.Tests.Builders.BuilderValue",
                entries_processed,
            )

        # Priority 10: Factory generation
        elif params.factory is not None:
            resolved_value = self._generate_from_factory(
                params.factory,
                params.count or c.Tests.Factory.DEFAULT_BATCH_COUNT,
            )

        # Priority 11: Model instantiation
        elif params.model is not None:
            data_dict = dict(params.model_data) if params.model_data else {}
            model_kind_str = self._get_model_kind(params.model)
            # Filter data_dict to only TestResultValue types
            filtered_dict: dict[str, t.Tests.TestResultValue] = {}
            for dict_key, dict_value in data_dict.items():
                if isinstance(dict_value, (str, int, float, bool, type(None))):
                    filtered_dict[dict_key] = cast(
                        "t.Tests.TestResultValue",
                        dict_value,
                    )
                elif u.is_type(dict_value, "list_or_tuple"):
                    # Cast to Sequence for list conversion (mypy needs explicit narrowing)
                    seq = cast("Sequence[t.GeneralValueType]", dict_value)
                    filtered_dict[dict_key] = cast(
                        "t.Tests.TestResultValue",
                        list(seq),
                    )
                elif u.is_type(dict_value, "dict"):
                    filtered_dict[dict_key] = cast(
                        "t.Tests.TestResultValue",
                        dict_value,
                    )
            # Cast to literal type for tt.model()
            model_kind: Literal["user", "config", "service", "entity", "value"] = cast(
                "Literal['user', 'config', 'service', 'entity', 'value']",
                model_kind_str,
            )
            resolved_value = cast(
                "t.Tests.Builders.BuilderValue",
                tt.model(model_kind, **filtered_dict),
            )

        # Priority 12: Config shortcuts
        elif params.production is not None or params.debug is not None:
            resolved_value = self._create_config(
                production=params.production or False,
                debug=params.debug
                if params.debug is not None
                else not (params.production or False),
            )

        # Priority 13: Mapping
        elif params.mapping is not None:
            # params.mapping is BuilderMapping which is Mapping[str, BuilderValue]
            mapping_dict = dict(
                cast("t.Tests.Builders.BuilderMapping", params.mapping),
            )
            resolved_value = cast("t.Tests.Builders.BuilderValue", mapping_dict)

        # Priority 14: Sequence
        elif params.sequence is not None:
            # params.sequence is BuilderSequence which is Sequence[BuilderValue]
            sequence_list = list(
                cast("t.Tests.Builders.BuilderSequence", params.sequence),
            )
            resolved_value = cast("t.Tests.Builders.BuilderValue", sequence_list)

        # Priority 15: Direct value
        elif params.value is not None:
            resolved_value = cast("t.Tests.Builders.BuilderValue", params.value)

        # Priority 16: Default
        elif params.default is not None:
            resolved_value = cast("t.Tests.Builders.BuilderValue", params.default)

        # Apply transformation if provided
        if params.transform is not None and resolved_value is not None:
            if u.is_type(resolved_value, "sequence"):
                # Cast to Sequence for iteration (pyrefly needs explicit narrowing)
                seq = cast("Sequence[t.GeneralValueType]", resolved_value)
                resolved_value = cast(
                    "t.Tests.Builders.BuilderValue",
                    [params.transform(item) for item in seq],
                )
            else:
                # Transform expects t.GeneralValueType, but BuilderValue is wider
                # Type narrowing: transform returns same type as input (BuilderValue)
                transformed = params.transform(
                    resolved_value,
                )
                resolved_value = transformed

        # Apply validation if provided
        if (
            params.validate_func is not None
            and resolved_value is not None
            and not params.validate_func(resolved_value)
        ):
            error_msg = (
                f"Validation failed for key '{params.key}' with value: {resolved_value}"
            )
            raise ValueError(error_msg)

        # Store result with merge support (use u.merge)
        self._ensure_data_initialized()
        # Type narrowing: _ensure_data_initialized() guarantees _data is not None
        assert self._data is not None, "_data must be initialized"
        builder_data: t.Tests.Builders.BuilderDict = self._data
        if params.merge and params.key in builder_data:
            existing = builder_data[params.key]
            if u.is_type(existing, "mapping") and u.is_type(resolved_value, "mapping"):
                # Cast to Mapping for dict conversion (pyrefly needs explicit narrowing)
                existing_map = cast("Mapping[str, t.GeneralValueType]", existing)
                resolved_map = cast("Mapping[str, t.GeneralValueType]", resolved_value)
                merge_result = u.merge(
                    dict(existing_map),
                    dict(resolved_map),
                )
                if merge_result.is_success:
                    resolved_value = cast(
                        "t.Tests.Builders.BuilderValue",
                        merge_result.value,
                    )
            else:
                builder_data[params.key] = resolved_value
        else:
            builder_data[params.key] = resolved_value
        # Update instance attribute after local modifications
        self._data = builder_data

        return self

    def set(
        self,
        path: str,
        value: t.Tests.Builders.BuilderValue | None = None,
        *,
        create_parents: bool = True,
        **kwargs: str | float | bool,
    ) -> Self:
        """Set value at nested path using dot notation.

        Args:
            path: Dot-separated path (e.g., "config.database.host").
            value: Value to set at path.
            create_parents: Whether to create intermediate dicts.
            **kwargs: Additional values to set as mapping at path.

        Returns:
            Self for method chaining.

        Examples:
            # Simple path
            tb().set("config.debug", True)

            # With kwargs
            tb().set("settings", host="localhost", port=8080)

            # Nested creation
            tb().set("a.b.c.d", value=42)

        """
        self._ensure_data_initialized()
        # Type narrowing: _ensure_data_initialized() guarantees _data is not None
        assert self._data is not None, "_data must be initialized"
        # If kwargs provided, merge with value as mapping
        final_value: t.Tests.Builders.BuilderValue
        if kwargs:
            if value is None:
                final_value = cast("t.Tests.Builders.BuilderValue", dict(kwargs))
            elif u.is_type(value, "mapping"):
                # Cast to Mapping for dict conversion (pyrefly needs explicit narrowing)
                value_map = cast("Mapping[str, t.GeneralValueType]", value)
                merged: t.ConfigurationDict = dict(value_map)
                merged.update(kwargs)
                final_value = cast("t.Tests.Builders.BuilderValue", merged)
            else:
                final_value = cast("t.Tests.Builders.BuilderValue", dict(kwargs))
        else:
            final_value = value

        parts = path.split(".")
        if len(parts) == 1:
            # Type narrowing: _ensure_data_initialized() guarantees _data is not None
            assert self._data is not None, "_data must be initialized"
            self._data[path] = final_value
            return self

        # Navigate to parent
        # Type narrowing: _ensure_data_initialized() guarantees _data is not None
        assert self._data is not None, "_data must be initialized"
        current: t.Tests.Builders.BuilderDict = self._data
        for part in parts[:-1]:
            if part not in current:
                if create_parents:
                    current[part] = {}
                else:
                    error_msg = f"Path '{part}' not found in '{path}'"
                    raise KeyError(error_msg)
            next_val = current[part]
            if not u.is_type(next_val, "dict"):
                if create_parents:
                    current[part] = {}
                    next_val = current[part]
                else:
                    error_msg = f"Path '{part}' is not a dict in '{path}'"
                    raise TypeError(error_msg)
            current = cast("dict[str, t.Tests.Builders.BuilderValue]", next_val)

        # current is BuilderDict, which accepts BuilderValue
        current[parts[-1]] = final_value
        return self

    def get(
        self,
        path: str,
        default: T | None = None,
        *,
        as_type: type[T] | None = None,
    ) -> T | None:
        """Get value from path.

        Args:
            path: Dot-separated path.
            default: Default value if not found.
            as_type: Type to cast result to.

        Returns:
            Value at path or default.

        """
        self._ensure_data_initialized()
        parts = path.split(".")
        current: object = self._data

        for part in parts:
            if not u.is_type(current, "mapping"):
                return default
            # Cast to Mapping for key access (pyrefly needs explicit narrowing)
            current_map = cast("Mapping[str, t.Tests.Builders.BuilderValue]", current)
            if part not in current_map:
                return default
            current = current_map[part]

        if as_type is not None:
            return cast("T", current)
        return cast("T | None", current) if current is not None else default

    def to_result[T](
        self,
        **kwargs: object,  # Accept object to allow Callable, etc. - validated by ToResultParams
    ) -> (
        r[T]
        | r[t.Tests.Builders.BuilderDict]
        | r[BaseModel]
        | r[list[T]]
        | r[dict[str, T]]
        | T
    ):
        """Build data wrapped in FlextResult.

        Uses Pydantic 2 models for parameter validation and computation.
        All parameters are validated using m.Tests.Builders.ToResultParams model.

        Args:
            **kwargs: Result parameters (as_model, error, unwrap, etc.)

        Returns:
            FlextResult containing built data or model, or unwrapped value if unwrap=True.

        Examples:
            # Success result
            result = tb().add("x", 1).to_result()

            # With error code
            result = tb().to_result(error="Failed", error_code="E001")

            # Auto-unwrap for fixtures
            value = tb().add("x", 1).to_result(unwrap=True)  # Returns {"x": 1} directly

            # With generic class
            result = tb().add("x", 1).to_result(as_cls=MyClass, cls_args=(1,))

        """
        # Convert kwargs to validated model using FlextUtilities
        params_result = u.Model.from_kwargs(m.Tests.Builders.ToResultParams, **kwargs)
        if params_result.is_failure:
            error_msg = f"Invalid to_result() parameters: {params_result.error}"
            raise ValueError(error_msg)
        params = params_result.value

        if params.error is not None:
            return r[t.Tests.Builders.BuilderDict].fail(
                params.error,
                error_code=params.error_code,
                error_data=params.error_data,
            )

        data = self.build()

        if params.validate_func is not None and not params.validate_func(
            data,
        ):
            return r[t.Tests.Builders.BuilderDict].fail(
                "Validation failed",
                error_code=params.error_code,
                error_data=params.error_data,
            )

        # Apply transformation
        if params.map_fn is not None:
            transformed = params.map_fn(data)
            if params.unwrap:
                return cast("T", transformed)
            result_val: r[T] = r.ok(cast("T", transformed))
            return result_val

        # Generic class instantiation
        if params.as_cls is not None:
            args = params.cls_args or ()
            try:
                instance = params.as_cls(*args, **data)
                if params.unwrap:
                    return cast("T", instance)
                result_instance_val: r[T] = r.ok(cast("T", instance))
                return result_instance_val
            except Exception as exc:
                return r[T].fail(
                    str(exc),
                    error_code=params.error_code,
                    error_data=params.error_data,
                )

        # Model instantiation
        if params.as_model is not None:
            try:
                model_instance = params.as_model(**data)
                if params.unwrap:
                    return cast("T", model_instance)
                return r[BaseModel].ok(model_instance)
            except Exception as exc:
                return r[BaseModel].fail(
                    str(exc),
                    error_code=params.error_code,
                    error_data=params.error_data,
                )

        # Batch result types
        if params.as_list_result:
            # data is always BuilderDict (dict), extract values
            values = list(data.values())
            if params.unwrap:
                return cast("T", values)
            return cast("r[T]", r.ok(values))

        if params.as_dict_result:
            if params.unwrap:
                return cast("T", data)
            return cast("r[T]", r.ok(data))

        # Standard result
        result = r[t.Tests.Builders.BuilderDict].ok(data)
        if params.unwrap:
            if result.is_failure:
                msg = params.unwrap_msg or f"Failed to unwrap result: {result.error}"
                raise ValueError(msg)
            return cast("T", result.value)
        return result

    @overload
    def build(self) -> t.Tests.Builders.BuilderDict: ...

    @overload
    def build[T](self, *, as_model: type[T], **kwargs: t.GeneralValueType) -> T: ...

    @overload
    def build(
        self,
        *,
        as_list: bool,
    ) -> list[tuple[str, t.Tests.Builders.BuilderValue]]: ...

    @overload
    def build(self, *, keys_only: bool) -> list[str]: ...

    @overload
    def build(
        self,
        *,
        values_only: bool,
    ) -> list[t.Tests.Builders.BuilderValue]: ...

    @overload
    def build(
        self,
        *,
        as_parametrized: bool = ...,
    ) -> list[t.Tests.Builders.ParametrizedCase]: ...

    def build[T](
        self,
        **kwargs: object,
    ) -> (
        t.Tests.Builders.BuilderDict
        | BaseModel
        | list[tuple[str, t.Tests.Builders.BuilderValue]]
        | list[str]
        | list[t.Tests.Builders.BuilderValue]
        | list[t.Tests.Builders.ParametrizedCase]
        | T
    ):
        """Build the dataset with output type control.

        Uses Pydantic 2 models for parameter validation and computation.
        All parameters are validated using m.Tests.Builders.BuildParams model.

        Args:
            **kwargs: Build parameters (as_model, as_list, flatten, etc.)

        Returns:
            Built dataset in requested format.

        Examples:
            # Default dict
            data = tb().add("x", 1).build()

            # Parametrized for pytest
            cases = tb().add("test_id", "case_1").add("value", 42).build(as_parametrized=True)
            # [("case_1", {"test_id": "case_1", "value": 42})]

            # With validation
            data = tb().add("count", 5).build(validate_with=lambda d: d["count"] > 0)

            # With transformation
            doubled = tb().add("x", 1).build(map_result=lambda d: d["x"] * 2)

        """
        # Convert kwargs to validated model using FlextUtilities
        # Use from_kwargs for simple kwargs (all FlexibleValue compatible)
        params_result = u.Model.from_kwargs(m.Tests.Builders.BuildParams, **kwargs)
        if params_result.is_failure:
            error_msg = f"Invalid build() parameters: {params_result.error}"
            raise ValueError(error_msg)
        params = params_result.value

        self._ensure_data_initialized()
        # Type narrowing: _ensure_data_initialized() guarantees _data is not None
        assert self._data is not None, "_data must be initialized"
        data: t.Tests.Builders.BuilderDict = dict(self._data)

        if params.filter_none:
            data = {k: v for k, v in data.items() if v is not None}

        if params.flatten:
            # Use fallback flatten implementation
            data = self._flatten_dict(data)

        # Apply validation
        if params.validate_with is not None and not params.validate_with(data):
            error_msg = "Validation failed during build"
            raise ValueError(error_msg)

        # Apply assertion
        if params.assert_with is not None:
            params.assert_with(data)

        # Apply transformation
        if params.map_result is not None:
            return cast("T", params.map_result(data))

        if params.keys_only:
            return list(data.keys())

        if params.values_only:
            return list(data.values())

        if params.as_list:
            return list(data.items())

        if params.as_parametrized:
            test_id = str(data.get(params.parametrize_key, "default"))
            return [(test_id, data)]

        if params.as_model is not None:
            return params.as_model(**data)

        return data

    def reset(self) -> Self:
        """Reset builder state.

        Returns:
            Self for method chaining.

        """
        # Set attribute directly (no PrivateAttr needed, compatible with FlextService)
        self._data = {}
        return self

    # =========================================================================
    # CONVENIENCE METHODS (Legacy API - Kept for backward compatibility)
    # =========================================================================

    def with_users(self, count: int = 5) -> Self:
        """Add test users to builder.

        Args:
            count: Number of users to generate.

        Returns:
            Self for method chaining.

        Example:
            builder.with_users(3)  # Adds 3 test users
            data = builder.build()
            assert len(data["users"]) == 3

        """
        users: list[t.ConfigurationDict] = [
            {
                "id": f"user_{i}",
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "active": True,
            }
            for i in range(count)
        ]
        return self.add(
            "users",
            value=cast("t.Tests.Builders.BuilderValue", users),
        )

    def with_configs(self, *, production: bool = False) -> Self:
        """Add configuration to builder.

        Args:
            production: If True, use production settings; otherwise development.

        Returns:
            Self for method chaining.

        Example:
            builder.with_configs(production=True)
            data = builder.build()
            assert data["configs"]["environment"] == "production"

        """
        config: t.ConfigurationDict = {
            "environment": "production" if production else "development",
            "debug": not production,
            "service_type": "api",
            "timeout": 30,
        }
        return self.add(
            "configs",
            value=cast("t.Tests.Builders.BuilderValue", config),
        )

    def with_validation_fields(self, count: int = 5) -> Self:
        """Add validation test fields to builder.

        Args:
            count: Number of valid emails to generate.

        Returns:
            Self for method chaining.

        Example:
            builder.with_validation_fields(3)
            data = builder.build()
            assert len(data["validation_fields"]["valid_emails"]) == 3

        """
        validation_fields: t.ConfigurationDict = {
            "valid_emails": [f"user{i}@example.com" for i in range(count)],
            "invalid_emails": ["invalid", "no-at-sign.com", "@missing-local.com"],
            "valid_hostnames": ["example.com", "localhost"],
        }
        return self.add(
            "validation_fields",
            value=cast("t.Tests.Builders.BuilderValue", validation_fields),
        )

    # =========================================================================
    # BUILDER COMPOSITION & CLONING
    # =========================================================================

    def copy_builder(self) -> Self:
        """Create independent copy of builder state.

        Examples:
            # Create base builder
            base = tb().add("users", factory="users", count=5)

            # Create variations
            REDACTED_LDAP_BIND_PASSWORD_variant = base.copy_builder().add("role", "REDACTED_LDAP_BIND_PASSWORD")
            user_variant = base.copy_builder().add("role", "user")

        Returns:
            New builder instance with copied data.

        """
        self._ensure_data_initialized()
        # Type narrowing: _ensure_data_initialized() guarantees _data is not None
        assert self._data is not None, "_data must be initialized"
        new_builder = FlextTestsBuilders()
        # Set attribute directly (no PrivateAttr needed, compatible with FlextService)
        new_builder._data = dict(self._data)
        return cast("Self", new_builder)

    def fork(self, **updates: t.GeneralValueType) -> Self:
        """Copy and immediately add updates.

        Uses Pydantic 2 models for parameter validation.
        All updates are validated using m.Tests.Builders.AddParams model via add().

        Examples:
            base = tb().add("config", mapping={"env": "test"})

            # Fork with updates
            prod = base.fork(env="prod", debug=False)
            dev = base.fork(env="dev", debug=True)

        Args:
            **updates: Key-value pairs to add to copied builder (validated via add()).

        Returns:
            New builder instance with copied data and updates.

        """
        new_builder = self.copy_builder()
        # Add each update using add() method for validation
        # add() uses u.Model.from_kwargs() internally for validation
        # Cast to BuilderValue since add() accepts BuilderValue | None
        for key, value in updates.items():
            _ = new_builder.add(
                key,
                value=cast("t.Tests.Builders.BuilderValue", value),
            )  # add() returns Self for chaining
        return new_builder

    def merge_from(
        self,
        other: FlextTestsBuilders,
        *,
        strategy: str = "deep",
        exclude_keys: frozenset[str] | None = None,
    ) -> Self:
        """Merge data from another builder.

        Uses Pydantic 2 models for parameter validation.
        All parameters are validated using m.Tests.Builders.MergeFromParams model.
        Uses FlextUtilities.merge() internally.

        Examples:
            common = tb().add("base", 1)
            specific = tb().add("extra", 2)

            combined = common.copy_builder().merge_from(specific)
            # {"base": 1, "extra": 2}

        Args:
            other: Another builder to merge from.
            strategy: Merge strategy ("deep", "override", "append", etc.).
            exclude_keys: Set of keys to exclude from merge.

        Returns:
            Self for method chaining.

        """
        # Convert kwargs to validated model using FlextUtilities
        params_result = u.Model.from_kwargs(
            m.Tests.Builders.MergeFromParams,
            strategy=strategy,
            exclude_keys=exclude_keys,
        )
        if params_result.is_failure:
            error_msg = f"Invalid merge_from() parameters: {params_result.error}"
            raise ValueError(error_msg)
        params = params_result.value

        self._ensure_data_initialized()
        # Type narrowing: _ensure_data_initialized() guarantees _data is not None
        assert self._data is not None, "_data must be initialized"
        # Ensure other builder's _data is also initialized
        other._ensure_data_initialized()
        assert other._data is not None, "other._data must be initialized"
        other_data = dict(other._data)
        if params.exclude_keys:
            other_data = {
                k: v for k, v in other_data.items() if k not in params.exclude_keys
            }

        merge_result = u.merge(
            cast("t.ConfigurationDict", self._data),
            cast("t.ConfigurationDict", other_data),
            strategy=params.strategy,
        )
        if merge_result.is_success:
            # Set attribute directly (no PrivateAttr needed, compatible with FlextService)
            self._ensure_data_initialized()
            assert self._data is not None, "_data must be initialized"
            self._data = cast(
                "t.Tests.Builders.BuilderDict",
                merge_result.value,
            )
        return self

    # =========================================================================
    # BATCH OPERATIONS & SCENARIOS
    # =========================================================================

    def batch(
        self,
        key: str,
        scenarios: Sequence[tuple[str, t.GeneralValueType]],
        **kwargs: t.GeneralValueType,
    ) -> Self:
        """Build batch of test scenarios.

        Uses Pydantic 2 models for parameter validation and computation.
        All parameters are validated using m.Tests.Builders.BatchParams model.

        Args:
            key: Key to store batch under.
            scenarios: Sequence of (scenario_id, data) tuples.
            **kwargs: Additional parameters (as_results, with_failures)

        Returns:
            Self for method chaining.

        Examples:
            # Create test scenarios
            tb().batch("cases", [
                ("valid_email", "test@example.com"),
                ("invalid_email", "not-an-email"),
            ])

            # With result wrapping
            tb().batch("results", [
                ("success", 42),
                ("another", 100),
            ], as_results=True)

            # Mixed success/failure
            tb().batch("results", [("ok", 1)], with_failures=[("err", "failed")])

        """
        # Convert kwargs to validated model using FlextUtilities
        # u.Model.from_kwargs() accepts **kwargs: object - Pydantic 2 handles conversions automatically
        # scenarios is passed as keyword argument - Pydantic 2 will validate the type
        params_result = u.Model.from_kwargs(
            m.Tests.Builders.BatchParams,
            key=key,
            scenarios=scenarios,
            **kwargs,
        )
        if params_result.is_failure:
            error_msg = f"Invalid batch() parameters: {params_result.error}"
            raise ValueError(error_msg)
        params = params_result.value

        self._ensure_data_initialized()
        # Type narrowing: _ensure_data_initialized() guarantees _data is not None
        assert self._data is not None, "_data must be initialized"
        builder_data: t.Tests.Builders.BuilderDict = self._data
        batch_data: list[t.GeneralValueType | r[t.GeneralValueType]] = []
        for _scenario_id, scenario_data in params.scenarios:
            if params.as_results:
                batch_data.append(r[t.GeneralValueType].ok(scenario_data))
            else:
                batch_data.append(scenario_data)

        if params.with_failures:
            for _fail_id, fail_error in params.with_failures:
                batch_data.append(r[t.GeneralValueType].fail(fail_error))

        builder_data[params.key] = cast(
            "t.Tests.Builders.BuilderValue",
            batch_data,
        )
        # Update instance attribute after local modifications
        self._data = builder_data
        return self

    def scenarios(
        self,
        *cases: tuple[str, dict[str, t.Tests.Builders.BuilderValue]],
    ) -> list[t.Tests.Builders.ParametrizedCase]:
        """Build pytest.mark.parametrize compatible scenarios.

        Examples:
            # Define scenarios directly
            cases = tb().scenarios(
                ("test_valid", {"input": "hello", "expected": 5}),
                ("test_empty", {"input": "", "expected": 0}),
                ("test_unicode", {"input": "🎉", "expected": 1}),
            )

            @pytest.mark.parametrize("test_id,data", cases)
            def test_length(self, test_id, data):
                assert len(data["input"]) == data["expected"]

        Args:
            *cases: Variable number of (test_id, data) tuples.

        Returns:
            List of parametrized test cases.

        """
        return list(cases)

    # =========================================================================
    # NO CONVENIENCE METHODS - Following FLEXT patterns:
    # Use add() directly with parameters, no wrappers or shortcuts
    # =========================================================================

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _get_model_kind(self, model: type[BaseModel]) -> str:
        """Map Pydantic model class to factory kind string.

        Args:
            model: Pydantic model class.

        Returns:
            Factory kind string ("user", "config", "service", "entity", "value").

        """
        model_name = model.__name__.lower()
        if "user" in model_name:
            return "user"
        if "config" in model_name:
            return "config"
        if "service" in model_name:
            return "service"
        if "entity" in model_name:
            return "entity"
        if "value" in model_name:
            return "value"
        return "user"  # Default fallback

    def _generate_from_factory(
        self,
        factory: str,
        count: int,
    ) -> t.Tests.Builders.BuilderValue:
        """Generate data using factory methods.

        Args:
            factory: Factory type name.
            count: Number of items to generate.

        Returns:
            Generated data.

        """
        if factory == "users":
            users: list[m.Tests.Factory.User] = cast(
                "list[m.Tests.Factory.User]",
                tt.batch("user", count=count),
            )
            return [
                {
                    c.Tests.Builders.KEY_ID: user.id,
                    c.Tests.Builders.KEY_NAME: user.name,
                    c.Tests.Builders.KEY_EMAIL: user.email,
                    c.Tests.Builders.KEY_ACTIVE: user.active,
                }
                for user in users
            ]

        if factory == "configs":
            return self._create_config(production=False, debug=True)

        if factory == "services":
            services: list[dict[str, str]] = []
            for i in range(count):
                service = tt.create_service(name=f"service_{i}")
                services.append({
                    "id": service.id,
                    "name": service.name,
                    "type": service.type,
                    "status": service.status,
                })
            return services

        if factory == "results":
            # Generate list of success results with integers
            values = list(range(count))
            results = tt.results(values)
            return [
                {
                    "success": res.is_success,
                    "value": res.value if res.is_success else None,
                }
                for res in results
            ]

        error_msg = f"Unknown factory: {factory}"
        raise ValueError(error_msg)

    def _create_config(
        self,
        *,
        production: bool,
        debug: bool,
    ) -> t.Tests.Builders.BuilderValue:
        """Create configuration data.

        Args:
            production: Whether production mode.
            debug: Whether debug mode.

        Returns:
            Configuration data dict.

        """
        environment = (
            c.Tests.Builders.DEFAULT_ENVIRONMENT_PRODUCTION
            if production
            else c.Tests.Builders.DEFAULT_ENVIRONMENT_DEVELOPMENT
        )
        config_result = tt.model(
            "config",
            service_type=c.Tests.Factory.DEFAULT_SERVICE_TYPE,
            environment=environment,
            debug=debug,
            timeout=c.Tests.Factory.DEFAULT_TIMEOUT,
        )
        # Type narrowing: tt.model() returns union type, extract BaseModel
        # Extract BaseModel from union type (single model case)
        if isinstance(config_result, r):
            config_unwrapped = config_result.value
            if isinstance(config_unwrapped, BaseModel):
                config = config_unwrapped
            else:
                msg = f"Expected BaseModel from result, got {type(config_unwrapped)}"
                raise TypeError(msg)
        elif isinstance(config_result, BaseModel):
            config = config_result
        elif isinstance(config_result, list):
            # Single model case shouldn't return list, but handle gracefully
            if len(config_result) == 1 and isinstance(config_result[0], BaseModel):
                config = config_result[0]
            else:
                msg = f"Expected single BaseModel, got list with {len(config_result)} items"
                raise TypeError(msg)
        elif isinstance(config_result, dict):
            # Single model case shouldn't return dict, but handle gracefully
            if len(config_result) == 1:
                config_value = next(iter(config_result.values()))
                if isinstance(config_value, BaseModel):
                    config = config_value
                else:
                    msg = f"Expected BaseModel in dict, got {type(config_value)}"
                    raise TypeError(msg)
            else:
                msg = f"Expected single BaseModel, got dict with {len(config_result)} items"
                raise TypeError(msg)
        else:
            msg = f"Expected BaseModel, got {type(config_result)}"
            raise TypeError(msg)
        # Type narrowing: config is BaseModel, cast to Config for attribute access
        config_typed = cast("m.Tests.Factory.Config", config)
        b = c.Tests.Builders
        return {
            b.KEY_SERVICE_TYPE: config_typed.service_type,
            b.KEY_ENVIRONMENT: config_typed.environment,
            b.KEY_DEBUG: config_typed.debug,
            b.KEY_LOG_LEVEL: config_typed.log_level,
            b.KEY_TIMEOUT: config_typed.timeout,
            b.KEY_MAX_RETRIES: config_typed.max_retries,
            b.KEY_DATABASE_URL: b.DEFAULT_DATABASE_URL,
            b.KEY_MAX_CONNECTIONS: b.DEFAULT_MAX_CONNECTIONS,
        }

    def _flatten_dict(
        self,
        data: t.Tests.Builders.BuilderDict,
        parent_key: str = "",
        sep: str = ".",
    ) -> t.Tests.Builders.BuilderDict:
        """Flatten nested dict using dot notation keys.

        Args:
            data: Dict to flatten.
            parent_key: Parent key prefix.
            sep: Separator for keys.

        Returns:
            Flattened dict.

        """
        items: list[tuple[str, t.Tests.Builders.BuilderValue]] = []
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            # Check if value is a Mapping but not a BaseModel
            if u.is_type(value, "mapping") and not hasattr(value, "model_dump"):
                # Cast to Mapping for dict conversion (pyrefly needs explicit narrowing)
                value_map = cast("Mapping[str, t.GeneralValueType]", value)
                items.extend(
                    self._flatten_dict(
                        cast("t.Tests.Builders.BuilderDict", dict(value_map)),
                        new_key,
                        sep,
                    ).items(),
                )
            else:
                items.append((new_key, value))
        return dict(items)

    # =========================================================================
    # STATIC NAMESPACE: tb.Tests.* - DELEGATION PATTERN
    # =========================================================================

    class Tests:
        """Test-specific builder helpers under tb.Tests.*.

        Pattern matches: u.Tests.*, c.Tests.*, m.Tests.*

        CRITICAL: All methods DELEGATE to existing utilities!
        """

        class Result:
            """FlextResult building helpers - tb.Tests.Result.*.

            DELEGATES TO: tt.res(), tu.Tests.Result.*, r[T].*
            """

            @staticmethod
            def ok[T](value: T) -> r[T]:
                """Create success result - DELEGATES to tt.res()."""
                result = tt.res("ok", value=value)
                if u.is_type(result, "list"):
                    # Cast to list for indexing (pyrefly needs explicit narrowing)
                    result_list = cast("list[r[T]]", result)
                    return result_list[0] if result_list else r[T].ok(value)
                return cast("r[T]", result)

            @staticmethod
            def fail[T](
                error: str,
                code: str | None = None,
                data: t.ConfigurationDict | None = None,
            ) -> r[T]:
                """Create failure result - DELEGATES to tt.res()."""
                error_code = code or c.Errors.VALIDATION_ERROR
                result: r[t.GeneralValueType] | list[r[t.GeneralValueType]] = tt.res(
                    "fail",
                    error=error,
                    error_code=error_code,
                )
                if u.is_type(result, "list"):
                    # Cast to list for indexing (pyrefly needs explicit narrowing)
                    result_list = cast("list[r[t.GeneralValueType]]", result)
                    return cast(
                        "r[T]",
                        result_list[0]
                        if result_list
                        else r[T].fail(error, error_code=error_code),
                    )
                return cast("r[T]", result)

            @staticmethod
            def batch_ok[T](values: Sequence[T]) -> list[r[T]]:
                """Create batch of success results - DELEGATES to tt.results()."""
                return tt.results(values=list(values))

            @staticmethod
            def batch_fail[T](errors: Sequence[str]) -> list[r[T]]:
                """Create batch of failure results - DELEGATES to tt.results()."""
                return tt.results(values=[], errors=list(errors))

            @staticmethod
            def mixed[T](
                successes: Sequence[T],
                errors: Sequence[str],
            ) -> list[r[T]]:
                """Create mixed batch - DELEGATES to tt.results()."""
                return tt.results(values=list(successes), errors=list(errors))

            @staticmethod
            def all_success[T](results: Sequence[r[T]]) -> bool:
                """Check if all results are successful."""
                return all(res.is_success for res in results)

            @staticmethod
            def partition[T](
                results: Sequence[r[T]],
            ) -> tuple[list[T], list[str]]:
                """Partition results into successes and errors."""
                successes = [res.value for res in results if res.is_success]
                errors = [str(res.error) for res in results if res.is_failure]
                return successes, errors

            @staticmethod
            def assert_success[T](result: r[T]) -> T:
                """Assert success - DELEGATES to tu.Tests.Result."""
                return tu.Tests.Result.assert_success(cast("p.Result[T]", result))

            @staticmethod
            def assert_failure(result: r[t.GeneralValueType]) -> str:
                """Assert failure - DELEGATES to tu.Tests.Result."""
                return tu.Tests.Result.assert_failure(
                    cast("p.Result[t.GeneralValueType]", result)
                )

        class Batch:
            """Batch operations - tb.Tests.Batch.*.

            DELEGATES TO: tu.Tests.GenericHelpers.*, tu.Tests.TestCaseHelpers.*
            """

            @staticmethod
            def scenarios[T](
                *cases: tuple[str, T],
            ) -> list[tuple[str, T]]:
                """Create parametrized test cases."""
                return list(cases)

            @staticmethod
            def from_dict[T](
                mapping: dict[str, T],
            ) -> list[tuple[str, T]]:
                """Convert dict to parametrized cases."""
                return list(mapping.items())

            @staticmethod
            def parametrized(
                success_values: Sequence[t.GeneralValueType],
                failure_errors: Sequence[str],
            ) -> list[tuple[str, t.ConfigurationDict]]:
                """Create parametrized cases - DELEGATES to tu.Tests.GenericHelpers."""
                cases = tu.Tests.GenericHelpers.create_parametrized_cases(
                    success_values=list(success_values),
                    failure_errors=list(failure_errors),
                )
                # Convert to (test_id, data) format
                result: list[tuple[str, t.ConfigurationDict]] = []
                for i, (res, is_success, value, error) in enumerate(cases):
                    test_id = f"case_{i}"
                    data: t.ConfigurationDict = cast(
                        "t.ConfigurationDict",
                        {
                            "result": res,
                            "is_success": is_success,
                            "value": value,
                            "error": error,
                        },
                    )
                    result.append((test_id, data))
                return result

            @staticmethod
            def test_cases(
                operation: str,
                descriptions: Sequence[str],
                inputs: Sequence[t.ConfigurationDict],
                expected: Sequence[t.GeneralValueType],
            ) -> list[t.ConfigurationDict]:
                """Create batch test cases - DELEGATES to tu.Tests.TestCaseHelpers."""
                return tu.Tests.TestCaseHelpers.create_batch_operation_test_cases(
                    operation=operation,
                    descriptions=list(descriptions),
                    input_data_list=list(inputs),
                    expected_results=list(expected),
                )

        class Data:
            """Data generation helpers - tb.Tests.Data.*.

            DELEGATES TO: u.Collection.*, u.Mapper.*, tu.Tests.Factory.*
            """

            @staticmethod
            def dict(**kwargs: t.GeneralValueType) -> t.ConfigurationDict:
                """Create typed dictionary - Uses t.ConfigurationDict type."""
                return dict(kwargs)

            @staticmethod
            def merged(
                *dicts: Mapping[str, t.GeneralValueType],
            ) -> t.ConfigurationDict:
                """Merge dictionaries - DELEGATES to u.merge()."""
                result: t.ConfigurationDict = {}
                for d in dicts:
                    merge_result = u.merge(result, dict(d))
                    if merge_result.is_success:
                        result = merge_result.value
                return result

            @staticmethod
            def flatten(
                nested: Mapping[str, t.GeneralValueType],
                separator: str = ".",
            ) -> t.ConfigurationDict:
                """Flatten nested dict - uses manual implementation."""

                # Manual flatten since u.Collection.flatten may not exist
                def _flatten(
                    data: dict[str, t.GeneralValueType],
                    parent: str = "",
                ) -> dict[str, t.GeneralValueType]:
                    items: list[tuple[str, t.GeneralValueType]] = []
                    for key, value in data.items():
                        new_key = f"{parent}{separator}{key}" if parent else key
                        # Check if value is a Mapping but not a BaseModel
                        if isinstance(value, Mapping) and not hasattr(
                            value,
                            "model_dump",
                        ):
                            items.extend(_flatten(dict(value), new_key).items())
                        else:
                            items.append((new_key, value))
                    return dict(items)

                return _flatten(dict(nested))

            @staticmethod
            def transform[T, U](
                items: Sequence[T],
                func: Callable[[T], U],
            ) -> list[U]:
                """Transform items - DELEGATES to u.Collection.map()."""
                return u.Collection.map(list(items), func)

            @staticmethod
            def id() -> str:
                """Generate UUID - DELEGATES to tu.Tests.Factory."""
                return tu.Tests.Factory.generate_id()

            @staticmethod
            def short_id(length: int = 8) -> str:
                """Generate short ID - DELEGATES to tu.Tests.Factory."""
                return tu.Tests.Factory.generate_short_id(length)

        class Model:
            """Model creation helpers - tb.Tests.Model.*.

            DELEGATES TO: tt.model(), tt.batch(), tu.Tests.DomainHelpers.*
            """

            @staticmethod
            def user(**overrides: t.GeneralValueType) -> m.Tests.Factory.User:
                """Create user - DELEGATES to tt.model()."""
                # Filter only valid TestResultValue types
                filtered: dict[str, t.Tests.TestResultValue] = {}
                for key, value in overrides.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        filtered[key] = cast("t.Tests.TestResultValue", value)
                    elif u.is_type(value, "list_or_tuple"):
                        # Cast to Sequence for list conversion (mypy needs explicit narrowing)
                        seq = cast("Sequence[t.GeneralValueType]", value)
                        filtered[key] = cast(
                            "t.Tests.TestResultValue",
                            list(seq),
                        )
                    elif u.is_type(value, "dict"):
                        filtered[key] = cast("t.Tests.TestResultValue", value)
                return cast("m.Tests.Factory.User", tt.model("user", **filtered))

            @staticmethod
            def config(**overrides: t.GeneralValueType) -> m.Tests.Factory.Config:
                """Create config - DELEGATES to tt.model()."""
                # Filter only valid TestResultValue types
                filtered: dict[str, t.Tests.TestResultValue] = {}
                for key, value in overrides.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        filtered[key] = cast("t.Tests.TestResultValue", value)
                    elif u.is_type(value, "list_or_tuple"):
                        # Cast to Sequence for list conversion (mypy needs explicit narrowing)
                        seq = cast("Sequence[t.GeneralValueType]", value)
                        filtered[key] = cast(
                            "t.Tests.TestResultValue",
                            list(seq),
                        )
                    elif u.is_type(value, "dict"):
                        filtered[key] = cast("t.Tests.TestResultValue", value)
                return cast("m.Tests.Factory.Config", tt.model("config", **filtered))

            @staticmethod
            def service(**overrides: t.GeneralValueType) -> m.Tests.Factory.Service:
                """Create service - DELEGATES to tt.model()."""
                # Filter only valid TestResultValue types
                filtered: dict[str, t.Tests.TestResultValue] = {}
                for key, value in overrides.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        filtered[key] = cast("t.Tests.TestResultValue", value)
                    elif u.is_type(value, "list_or_tuple"):
                        # Cast to Sequence for list conversion (mypy needs explicit narrowing)
                        seq = cast("Sequence[t.GeneralValueType]", value)
                        filtered[key] = cast(
                            "t.Tests.TestResultValue",
                            list(seq),
                        )
                    elif u.is_type(value, "dict"):
                        filtered[key] = cast("t.Tests.TestResultValue", value)
                return cast("m.Tests.Factory.Service", tt.model("service", **filtered))

            @staticmethod
            def entity[T: m.Entity](
                entity_class: type[T],
                name: str = "",
                value: t.GeneralValueType = "",
            ) -> T:
                """Create entity - DELEGATES to tu.Tests.DomainHelpers."""
                return tu.Tests.DomainHelpers.create_test_entity_instance(
                    name=name,
                    value=value,
                    entity_class=entity_class,
                )

            @staticmethod
            def value_object[T: m.Value](
                value_class: type[T],
                data: str = "",
                count: int = 1,
            ) -> T:
                """Create value object - DELEGATES to tu.Tests.DomainHelpers."""
                return tu.Tests.DomainHelpers.create_test_value_object_instance(
                    data=data,
                    count=count,
                    value_class=value_class,
                )

            @staticmethod
            def batch_users(
                count: int = c.Tests.Factory.DEFAULT_BATCH_COUNT,
            ) -> list[m.Tests.Factory.User]:
                """Create batch users - DELEGATES to tt.batch()."""
                return cast("list[m.Tests.Factory.User]", tt.batch("user", count=count))

            @staticmethod
            def batch_entities[T: m.Entity](
                entity_class: type[T],
                names: Sequence[str],
                values: Sequence[t.GeneralValueType],
            ) -> list[T]:
                """Create batch entities - DELEGATES to tu.Tests.DomainHelpers."""
                result = tu.Tests.DomainHelpers.create_test_entities_batch(
                    names=list(names),
                    values=list(values),
                    entity_class=entity_class,
                )
                if result.is_success:
                    return result.value
                raise ValueError(result.error)

        class Operation:
            """Operation helpers - tb.Tests.Operation.*.

            DELEGATES TO: tu.Tests.Factory.*, tt.op()
            """

            @staticmethod
            def simple() -> Callable[[], str]:
                """Simple operation - DELEGATES to tu.Tests.Factory."""
                return cast("Callable[[], str]", tu.Tests.Factory.simple_operation)

            @staticmethod
            def add() -> Callable[[int, int], int]:
                """Add operation - DELEGATES to tu.Tests.Factory."""
                return cast("Callable[[int, int], int]", tu.Tests.Factory.add_operation)

            @staticmethod
            def format() -> Callable[[str, int], str]:
                """Format operation - DELEGATES to tu.Tests.Factory."""
                return cast(
                    "Callable[[str, int], str]",
                    tu.Tests.Factory.format_operation,
                )

            @staticmethod
            def error(message: str) -> Callable[[], None]:
                """Error operation - DELEGATES to tu.Tests.Factory."""
                return cast(
                    "Callable[[], None]",
                    tu.Tests.Factory.create_error_operation(message),
                )

            @staticmethod
            def execute_service(
                overrides: t.ConfigurationDict | None = None,
            ) -> r[t.GeneralValueType]:
                """Execute service - DELEGATES to tu.Tests.Factory."""
                return tu.Tests.Factory.execute_user_service(overrides or {})


# Short alias for convenient test usage
tb = FlextTestsBuilders

__all__ = ["FlextTestsBuilders", "tb"]
