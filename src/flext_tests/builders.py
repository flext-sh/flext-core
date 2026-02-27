"""Test data builders for FLEXT ecosystem tests.

Provides ultra-powerful builder pattern for creating complex test data structures.
Supports FlextResult, lists, dicts, mappings, and generic classes with fluent interface.

Uses flext-core utilities extensively to avoid code duplication.
Designed with minimal public methods that handle almost everything.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Literal, Self, TypeGuard, overload

from flext_core import r
from pydantic import BaseModel

from flext_tests.constants import c
from flext_tests.factories import tt
from flext_tests.models import m
from flext_tests.typings import t
from flext_tests.utilities import u


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
    # CRITICAL: Every instance __init__ MUST set _data to a dict
    _data: t.Tests.Builders.BuilderDict

    @staticmethod
    def _is_result_obj(value: object) -> TypeGuard[r[t.Tests.PayloadValue]]:
        return isinstance(value, r)

    @staticmethod
    def _to_payload_value(value: object) -> t.Tests.PayloadValue:
        if value is None or isinstance(
            value, str | int | float | bool | bytes | BaseModel
        ):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): FlextTestsBuilders._to_payload_value(item)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [FlextTestsBuilders._to_payload_value(item) for item in value]
        return str(value)

    @staticmethod
    def _to_guard_input(value: t.Tests.PayloadValue) -> t.GuardInputValue:
        if value is None or isinstance(value, str | int | float | bool | BaseModel):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): FlextTestsBuilders._to_guard_input(
                    FlextTestsBuilders._to_payload_value(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, bytes):
            return str(value)
        return [
            FlextTestsBuilders._to_guard_input(
                FlextTestsBuilders._to_payload_value(item)
            )
            for item in value
        ]

    def __init__(self, **data: t.Tests.PayloadValue) -> None:
        """Initialize builder with optional initial data."""
        super().__init__()
        # Initialize without inheritance
        # Set attribute directly (no PrivateAttr needed, compatible with FlextService)
        # Always initialize as empty dict (class attribute is default, instance needs fresh copy)
        self._data = dict(data)

    def _ensure_data_initialized(self) -> None:
        """Ensure _data is initialized (helper for type safety)."""
        if not self._data:
            self._data = {}

    def execute(self) -> r[t.Tests.Builders.BuilderDict]:
        """Execute service - builds and returns as FlextResult.

        Returns:
            FlextResult containing built data.

        """
        self._ensure_data_initialized()
        assert self._data is not None, "_data must be initialized"
        data: t.Tests.Builders.BuilderDict = dict(self._data)
        return r[t.Tests.Builders.BuilderDict].ok(data)

    # =========================================================================
    # CORE PUBLIC METHODS - Minimal and Powerful
    # =========================================================================

    def add(
        self,
        key: str,
        value: t.Tests.Builders.BuilderValue | None = None,
        **kwargs: t.Tests.PayloadValue,  # Accept payload values - validated by AddParams
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
        # Convert value to payload value for storage
        # Builders store DATA only - use to_result() for FlextResult
        value_for_kwargs: t.Tests.PayloadValue = None

        if value is None:
            value_for_kwargs = None
        elif type(value) in {str, int, float, bool} or BaseModel in type(value).__mro__:
            value_for_kwargs = value
        elif isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            value_for_kwargs = list(value)
        elif isinstance(value, dict):
            value_for_kwargs = dict(value)
        else:
            # Other types: convert to string representation
            value_for_kwargs = str(value)

        try:
            params = m.Tests.Builders.AddParams.model_validate(
                {"key": key, "value": value_for_kwargs, **kwargs},
            )
        except (TypeError, ValueError, AttributeError) as exc:
            error_msg = f"Invalid add() parameters: {exc}"
            raise ValueError(error_msg) from exc

        resolved_value: t.Tests.PayloadValue = None

        # Priority 2: FlextResult success (result_ok)
        # Store as data with metadata - result wrapping happens at to_result() time
        if params.result_ok is not None:
            resolved_value = {
                "_result_ok": params.result_ok,
                "_is_result_marker": True,
            }

        # Priority 3: FlextResult failure (result_fail)
        # Store as data with metadata - result wrapping happens at to_result() time
        elif params.result_fail is not None:
            result_code = params.result_code or c.Errors.VALIDATION_ERROR
            resolved_value = {
                "_result_fail": params.result_fail,
                "_result_code": result_code,
                "_is_result_marker": True,
            }

        # Priority 7: Generic class instantiation
        elif params.cls is not None:
            args = params.cls_args or ()
            cls_kwargs = params.cls_kwargs or {}
            # Check if it's a domain entity or value object
            cls_type = params.cls

            # Type guards for Entity and Value classes
            def is_entity_class(
                cls: type[object],
            ) -> TypeGuard[type[m.Tests.Factory.Entity]]:
                """Type guard to check if class is Entity subclass."""
                return issubclass(cls, m.Tests.Factory.Entity)

            def is_value_class(
                cls: type[object],
            ) -> TypeGuard[type[m.Tests.Factory.Value]]:
                """Type guard to check if class is Value subclass."""
                return issubclass(cls, m.Tests.Factory.Value)

            # Type narrowing for Entity classes
            if is_entity_class(cls_type):
                entity_cls: type[m.Tests.Factory.Entity] = cls_type
                name_val = cls_kwargs.get("name", "")
                value_val = cls_kwargs.get("value", "")

                def entity_factory(
                    *,
                    name: str,
                    value: t.Tests.PayloadValue,
                ) -> m.Tests.Factory.Entity:
                    return entity_cls(name=name, value=value)

                resolved_value = u.Tests.DomainHelpers.create_test_entity_instance(
                    name=str(name_val) if name_val else "",
                    value=value_val or "",
                    entity_class=entity_factory,
                )
            # Type narrowing for Value classes
            elif is_value_class(cls_type):
                value_cls: type[m.Tests.Factory.Value] = cls_type
                data_val = cls_kwargs.get("data", "")
                count_val = cls_kwargs.get("count", 1)

                def value_factory(
                    *,
                    data: str,
                    count: int,
                ) -> m.Tests.Factory.Value:
                    return value_cls(data=data, count=count)

                resolved_value = (
                    u.Tests.DomainHelpers.create_test_value_object_instance(
                        data=str(data_val) if data_val else "",
                        count=int(count_val)
                        if isinstance(count_val, int | float)
                        else 1,
                        value_class=value_factory,
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
                # Type narrow: dynamic class instantiation results are BuilderValue
                if isinstance(
                    instance,
                    str | int | float | bool | list | dict | BaseModel | type(None),
                ):
                    resolved_value = instance
                else:
                    # Fallback: store as-is, Pydantic handles validation
                    resolved_value = str(instance) if instance is not None else None

        # Priority 8: Items with transformation/filtering
        elif params.items is not None:
            items_processed = list(params.items)
            if params.items_filter is not None:
                items_processed = [
                    item for item in items_processed if params.items_filter(item)
                ]
            if params.items_map is not None:
                items_processed = [params.items_map(item) for item in items_processed]
            resolved_value = items_processed

        # Priority 9: Entries with transformation/filtering
        elif params.entries is not None:
            entries_processed: dict[str, t.Tests.PayloadValue] = dict(params.entries)
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
            resolved_value = entries_processed

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
                if type(dict_value) in {str, int, float, bool, type(None)}:
                    filtered_dict[dict_key] = dict_value
                elif isinstance(dict_value, Sequence) and not isinstance(
                    dict_value, str | bytes | bytearray
                ):
                    filtered_dict[dict_key] = list(dict_value)
                elif isinstance(dict_value, dict):
                    filtered_dict[dict_key] = dict_value
            # model_kind_str is validated by _get_model_kind - use match for type safety
            model_kind: Literal["user", "config", "service", "entity", "value"]
            match model_kind_str:
                case "user" | "config" | "service" | "entity" | "value":
                    model_kind = model_kind_str
                case _:
                    model_kind = "user"  # Default fallback
            # tt.model returns BaseModel which is compatible with BuilderValue
            model_result = tt.model(model_kind, **filtered_dict)
            # Type narrow - handle all return type variants
            if isinstance(model_result, BaseModel):
                resolved_value = model_result
            elif self._is_result_obj(model_result):
                if model_result.is_success:
                    result_val = model_result.value
                    if isinstance(result_val, BaseModel):
                        resolved_value = result_val
                    elif isinstance(result_val, Sequence) and not isinstance(
                        result_val,
                        str | bytes,
                    ):
                        resolved_value = list(result_val)
                    elif isinstance(result_val, Mapping):
                        resolved_value = dict(result_val)
                else:
                    resolved_value = None
            elif isinstance(model_result, list | dict):
                resolved_value = model_result

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
            # params.mapping is already validated - convert to dict
            resolved_value = dict(params.mapping)

        # Priority 14: Sequence
        elif params.sequence is not None:
            # params.sequence is already validated - convert to list
            resolved_value = list(params.sequence)

        # Priority 15: Direct value
        elif params.value is not None:
            # params.value is payload-compatible and subset of BuilderValue
            resolved_value = params.value

        # Priority 16: Default
        elif params.default is not None:
            # params.default is payload-compatible and subset of BuilderValue
            resolved_value = params.default

        # Apply transformation if provided
        if params.transform is not None and resolved_value is not None:
            if isinstance(resolved_value, Sequence) and not isinstance(
                resolved_value,
                str | bytes,
            ):
                resolved_value = [params.transform(item) for item in resolved_value]
            else:
                resolved_value = params.transform(resolved_value)

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
            if isinstance(existing, Mapping) and isinstance(resolved_value, Mapping):
                # Both are Mapping - convert to dict[str, payload value] for merge
                # Filter out FlextResult values for merge compatibility
                existing_dict: dict[str, t.Tests.PayloadValue] = {}
                for k, v in existing.items():
                    if not self._is_result_obj(v):
                        existing_dict[str(k)] = v
                resolved_dict: dict[str, t.Tests.PayloadValue] = {}
                for k, v in resolved_value.items():
                    if not self._is_result_obj(v):
                        resolved_dict[str(k)] = v
                merge_result = u.merge(
                    {
                        key: self._to_guard_input(val)
                        for key, val in existing_dict.items()
                    },
                    {
                        key: self._to_guard_input(val)
                        for key, val in resolved_dict.items()
                    },
                )
                if merge_result.is_success:
                    resolved_value = self._to_payload_value(merge_result.value)
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
        **kwargs: t.Tests.PayloadValue,
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
            tb().set("settings", host=c.Platform.DEFAULT_HOST, port=c.Platform.DEFAULT_HTTP_PORT)

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
                final_value = dict(kwargs)
            elif isinstance(value, Mapping):
                merged: dict[str, t.Tests.PayloadValue] = dict(value)
                merged.update(kwargs)
                final_value = merged
            else:
                final_value = dict(kwargs)
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
            if not isinstance(next_val, dict):
                if create_parents:
                    current[part] = {}
                    next_val = current[part]
                else:
                    error_msg = f"Path '{part}' is not a dict in '{path}'"
                    raise TypeError(error_msg)
            if isinstance(next_val, dict):
                current = next_val

        # current is BuilderDict, which accepts BuilderValue
        current[parts[-1]] = final_value
        return self

    @overload
    def get(self, path: str) -> t.Tests.Builders.BuilderValue | None: ...

    @overload
    def get[T](
        self,
        path: str,
        default: T,
    ) -> t.Tests.Builders.BuilderValue | T: ...

    @overload
    def get[T](
        self,
        path: str,
        default: T | None = None,
        *,
        as_type: type[T],
    ) -> T | None: ...

    def get[T](
        self,
        path: str,
        default: T | None = None,
        *,
        as_type: type[T] | None = None,
    ) -> t.Tests.Builders.BuilderValue | T | None:
        """Get value from path.

        Args:
            path: Dot-separated path.
            default: Default value if not found.
            as_type: Type to validate and narrow result to.

        Returns:
            Value at path or default.

        """
        self._ensure_data_initialized()
        parts = path.split(".")
        current: t.ConfigMapValue = self._data

        for part in parts:
            if not isinstance(current, Mapping):
                return default
            current_mapping = current
            if part not in current_mapping:
                return default
            current = current_mapping[part]

        # Return the value
        if current is None:
            return default
        # For as_type, type check provides narrowing
        if as_type is not None:
            if isinstance(current, as_type):
                typed_current: T = current
                return typed_current
            return default
        if isinstance(current, str | int | float | bool | bytes | BaseModel):
            return current
        if isinstance(current, Mapping):
            return {str(k): self._to_payload_value(v) for k, v in current.items()}
        if isinstance(current, Sequence):
            return [self._to_payload_value(item) for item in current]
        return default

    def to_result(
        self,
        **kwargs: t.Tests.PayloadValue,  # Accept payload values - validated by ToResultParams
    ) -> r[t.Tests.Builders.BuilderValue] | t.Tests.Builders.BuilderValue:
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
        try:
            params = m.Tests.Builders.ToResultParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            error_msg = f"Invalid to_result() parameters: {exc}"
            raise ValueError(error_msg) from exc

        if params.error is not None:
            return r[t.Tests.PayloadValue].fail(
                params.error,
                error_code=params.error_code,
                error_data=params.error_data,
            )

        # Get data directly from internal storage (not from build() to preserve type)
        self._ensure_data_initialized()
        assert self._data is not None, "_data must be initialized"
        data: t.Tests.Builders.BuilderDict = dict(self._data)

        if params.validate_func is not None and not params.validate_func(data):
            return r[t.Tests.PayloadValue].fail(
                "Validation failed",
                error_code=params.error_code,
                error_data=params.error_data,
            )

        # Apply transformation
        if params.map_fn is not None:
            transformed = params.map_fn(data)
            if params.unwrap:
                if t.Guards.is_builder_value(transformed):
                    return transformed
                return None
            return r[t.Tests.PayloadValue].ok(transformed)

        # Generic class instantiation
        if params.as_cls is not None:
            args = params.cls_args or ()
            try:
                instance = params.as_cls(*args, **data)
                if params.unwrap:
                    if t.Guards.is_builder_value(instance):
                        return instance
                    return None
                if t.Guards.is_builder_value(instance):
                    return r[t.Tests.PayloadValue].ok(instance)
                return r[t.Tests.PayloadValue].ok(None)
            except (TypeError, ValueError, AttributeError) as exc:
                return r[t.Tests.PayloadValue].fail(
                    str(exc),
                    error_code=params.error_code,
                    error_data=params.error_data,
                )

        # Model instantiation
        if params.as_model is not None:
            try:
                model_instance = params.as_model(**data)
                if params.unwrap:
                    return model_instance
                return r[t.Tests.PayloadValue].ok(model_instance)
            except (TypeError, ValueError, AttributeError) as exc:
                return r[t.Tests.PayloadValue].fail(
                    str(exc),
                    error_code=params.error_code,
                    error_data=params.error_data,
                )

        # Batch result types
        if params.as_list_result:
            values: list[t.Tests.PayloadValue] = list(data.values())
            if params.unwrap:
                return values
            return r[t.Tests.PayloadValue].ok(values)

        if params.as_dict_result:
            if params.unwrap:
                return data
            # Widen dict to payload value for type compatibility
            dict_as_value: t.Tests.PayloadValue = data
            return r[t.Tests.PayloadValue].ok(dict_as_value)

        # Standard result
        # Widen dict to payload value for type compatibility
        data_as_value: t.Tests.PayloadValue = data
        result: r[t.Tests.Builders.BuilderValue] = r[t.Tests.Builders.BuilderValue].ok(
            data_as_value,
        )
        if params.unwrap:
            if result.is_failure:
                msg = params.unwrap_msg or f"Failed to unwrap result: {result.error}"
                raise ValueError(msg)
            return result.value
        return result

    def build(
        self,
        **kwargs: t.Tests.PayloadValue,  # Accept payload values - validated by BuildParams
    ) -> Self:
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
        # Use from_kwargs for simple kwargs (all payload-compatible)
        try:
            params = m.Tests.Builders.BuildParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            error_msg = f"Invalid build() parameters: {exc}"
            raise ValueError(error_msg) from exc

        self._ensure_data_initialized()
        # Type narrowing: _ensure_data_initialized() guarantees _data is not None
        assert self._data is not None, "_data must be initialized"
        builder_data: t.Tests.Builders.BuilderDict = dict(self._data)

        # Convert batch results with _is_result or _is_failure flags to FlextResult
        data = self._process_batch_results(builder_data)

        if params.filter_none:
            data = {k: v for k, v in data.items() if v is not None}

        if params.flatten:
            # Use fallback flatten implementation - needs BuilderDict input
            flat_input: t.Tests.Builders.BuilderDict = {}
            for k, v in data.items():
                if not isinstance(v, r):
                    flat_input[k] = self._to_payload_value(v)
            data = dict(self._flatten_dict(flat_input))

        # Apply validation
        data_for_hooks: t.Tests.Builders.BuilderOutputDict = {
            str(key): value for key, value in data.items()
        }
        if params.validate_with is not None and not params.validate_with(
            data_for_hooks
        ):
            error_msg = "Validation failed during build"
            raise ValueError(error_msg)

        # Apply assertion
        if params.assert_with is not None:
            params.assert_with(data_for_hooks)

        # Apply transformation
        if params.map_result is not None:
            return params.map_result(data_for_hooks)

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
            model_kwargs: dict[str, t.Tests.PayloadValue] = {
                k: self._to_payload_value(v) for k, v in data.items()
            }
            return params.as_model(**model_kwargs)

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
        users: list[dict[str, t.Tests.PayloadValue]] = [
            {
                "id": f"user_{i}",
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "active": True,
            }
            for i in range(count)
        ]
        # list[ConfigurationDict] is compatible with BuilderValue
        return self.add("users", value=users)

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
        config: dict[str, t.Tests.PayloadValue] = {
            "environment": "production" if production else "development",
            "debug": not production,
            "service_type": "api",
            "timeout": 30,
        }
        # ConfigurationDict is compatible with BuilderValue
        return self.add("configs", value=config)

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
        validation_fields: dict[str, t.Tests.PayloadValue] = {
            "valid_emails": [f"user{i}@example.com" for i in range(count)],
            "invalid_emails": ["invalid", "no-at-sign.com", "@missing-local.com"],
            "valid_hostnames": ["example.com", c.Platform.DEFAULT_HOST],
        }
        # ConfigurationDict is compatible with BuilderValue
        return self.add("validation_fields", value=validation_fields)

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
        # Create new instance of same class (for proper subclass support)
        new_builder = type(self)()
        # Set attribute directly (no PrivateAttr needed, compatible with FlextService)
        new_builder._data = dict(self._data)
        return new_builder

    def fork(self, **updates: t.Tests.PayloadValue) -> Self:
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
        # Payload value is compatible with BuilderValue (it's a subset)
        for key, value in updates.items():
            _ = new_builder.add(key, value=value)
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
        # Convert frozenset to list for payload compatibility
        # Pydantic 2 will coerce list back to frozenset in the model
        try:
            params = m.Tests.Builders.MergeFromParams.model_validate(
                {
                    "strategy": strategy,
                    "exclude_keys": list(exclude_keys) if exclude_keys else None,
                },
            )
        except (TypeError, ValueError, AttributeError) as exc:
            error_msg = f"Invalid merge_from() parameters: {exc}"
            raise ValueError(error_msg) from exc

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

        # Convert BuilderDict to ConfigurationDict for merge
        # Use TypeGuard t.Guards.is_general_value for proper type narrowing
        self_dict: dict[str, t.Tests.PayloadValue] = {
            k: v for k, v in self._data.items() if t.Guards.is_general_value(v)
        }
        other_dict: dict[str, t.Tests.PayloadValue] = {
            k: v for k, v in other_data.items() if t.Guards.is_general_value(v)
        }
        merge_result = u.merge(
            {k: self._to_guard_input(v) for k, v in self_dict.items()},
            {k: self._to_guard_input(v) for k, v in other_dict.items()},
            strategy=params.strategy,
        )
        if merge_result.is_success:
            self._ensure_data_initialized()
            # _data is always a dict after __init__ and _ensure_data_initialized
            for k, v in merge_result.value.items():
                self._data[k] = self._to_payload_value(v)
        return self

    # =========================================================================
    # BATCH OPERATIONS & SCENARIOS
    # =========================================================================

    def batch(
        self,
        key: str,
        scenarios: Sequence[tuple[str, t.Tests.PayloadValue]],
        **kwargs: t.Tests.PayloadValue,
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
        try:
            params = m.Tests.Builders.BatchParams.model_validate(
                {"key": key, "scenarios": scenarios, **kwargs},
            )
        except (TypeError, ValueError, AttributeError) as exc:
            error_msg = f"Invalid batch() parameters: {exc}"
            raise ValueError(error_msg) from exc

        self._ensure_data_initialized()
        # Type narrowing: _ensure_data_initialized() guarantees _data is not None
        assert self._data is not None, "_data must be initialized"
        builder_data: t.Tests.Builders.BuilderDict = self._data
        # Builders store DATA only - as_results wrapping happens when accessed
        batch_data: list[t.Tests.PayloadValue] = []
        for scenario_id, scenario_data in params.scenarios:
            # Store as dict with id and data - results wrapped later
            if params.as_results:
                batch_data.append(
                    {
                        "_id": scenario_id,
                        "_result_ok": scenario_data,
                        "_is_result_marker": True,
                    },
                )
            else:
                batch_data.append(scenario_data)

        if params.with_failures:
            for fail_id, fail_error in params.with_failures:
                # Store failure markers - results wrapped later
                batch_data.append(
                    {
                        "_id": fail_id,
                        "_result_fail": fail_error,
                        "_is_result_marker": True,
                    },
                )

        # list[payload value] is compatible with BuilderValue
        builder_data[params.key] = batch_data
        # Update instance attribute after local modifications
        self._data = builder_data
        return self

    def scenarios(
        self,
        *cases: tuple[str, Mapping[str, t.Tests.Builders.BuilderValue]],
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
        msg = f"Unknown model kind for {model.__name__}"
        raise ValueError(msg)

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
            batch_result = tt.batch("user", count=count)
            users_data: list[dict[str, t.Tests.PayloadValue]] = [
                {
                    c.Tests.Builders.KEY_ID: item.id,
                    c.Tests.Builders.KEY_NAME: item.name,
                    c.Tests.Builders.KEY_EMAIL: item.email,
                    c.Tests.Builders.KEY_ACTIVE: item.active,
                }
                for item in batch_result
                if isinstance(item, m.Tests.Factory.User)
            ]
            return users_data

        if factory == "configs":
            return self._create_config(production=False, debug=True)

        if factory == "services":
            services: list[dict[str, str]] = []
            for i in range(count):
                service_result = tt.model("service", name=f"service_{i}")
                if not isinstance(service_result, m.Tests.Factory.Service):
                    continue
                services.append({
                    "id": service_result.id,
                    "name": service_result.name,
                    "type": service_result.type,
                    "status": service_result.status,
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
        # Single path: extract BaseModel from tt.model() result
        config: m.Tests.Factory.Config
        if self._is_result_obj(config_result):
            config_unwrapped = config_result.value
            if isinstance(config_unwrapped, m.Tests.Factory.Config):
                config = config_unwrapped
            else:
                msg = f"Expected Config from result, got {type(config_unwrapped)}"
                raise TypeError(msg)
        elif isinstance(config_result, m.Tests.Factory.Config):
            config = config_result
        elif isinstance(config_result, list):
            if len(config_result) == 1 and BaseModel in type(config_result[0]).__mro__:
                only_item = config_result[0]
                if isinstance(only_item, m.Tests.Factory.Config):
                    config = only_item
                else:
                    msg = f"Expected Config model, got {type(only_item)}"
                    raise TypeError(msg)
            else:
                msg = f"Expected single BaseModel, got list with {len(config_result)} items"
                raise TypeError(msg)
        elif isinstance(config_result, dict):
            if len(config_result) == 1:
                config_value = next(iter(config_result.values()))
                if isinstance(config_value, m.Tests.Factory.Config):
                    config = config_value
                else:
                    msg = f"Expected Config in dict result, got {type(config_value)}"
                    raise TypeError(msg)
            else:
                msg = f"Expected single BaseModel, got dict with {len(config_result)} items"
                raise TypeError(msg)
        else:
            msg = f"Expected BaseModel from config_result, got {type(config_result)}"
            raise TypeError(msg)
        b = c.Tests.Builders
        config_data: dict[str, t.Tests.PayloadValue] = {
            b.KEY_SERVICE_TYPE: config.service_type,
            b.KEY_ENVIRONMENT: config.environment,
            b.KEY_DEBUG: config.debug,
            b.KEY_LOG_LEVEL: config.log_level,
            b.KEY_TIMEOUT: config.timeout,
            b.KEY_MAX_RETRIES: config.max_retries,
            b.KEY_DATABASE_URL: b.DEFAULT_DATABASE_URL,
            b.KEY_MAX_CONNECTIONS: b.DEFAULT_MAX_CONNECTIONS,
        }
        return config_data

    def _process_batch_results(
        self,
        data: t.Tests.Builders.BuilderDict,
    ) -> t.Tests.Builders.BuilderOutputDict:
        """Convert batch result markers to actual FlextResult objects.

        Processes lists containing dicts with _is_result or _is_failure flags
        and converts them to r[T].ok() or r[T].fail() instances.

        Args:
            data: Builder data dict to process.

        Returns:
            Processed data with FlextResult objects.

        """
        processed: dict[
            str,
            (
                t.Tests.PayloadValue
                | r[t.Tests.PayloadValue]
                | list[t.Tests.PayloadValue | r[t.Tests.PayloadValue]]
                | Mapping[str, t.Tests.PayloadValue]
            ),
        ] = {}
        for key, value in data.items():
            if isinstance(value, list):
                converted_items: list[
                    t.Tests.PayloadValue | r[t.Tests.PayloadValue]
                ] = []
                for item in value:
                    if isinstance(item, dict) and item.get("_is_result_marker"):
                        if "_result_ok" in item:
                            converted_items.append(
                                r[t.Tests.PayloadValue].ok(item["_result_ok"]),
                            )
                        elif "_result_fail" in item:
                            error_msg = str(item["_result_fail"])
                            converted_items.append(
                                r[t.Tests.PayloadValue].fail(error_msg),
                            )
                        else:
                            converted_items.append(item)
                    else:
                        converted_items.append(item)
                processed[key] = converted_items
            elif isinstance(value, dict) and value.get("_is_result_marker"):
                # Handle add() with result_ok or result_fail
                if "_result_ok" in value:
                    processed[key] = r[t.Tests.PayloadValue].ok(value["_result_ok"])
                elif "_result_fail" in value:
                    error_msg = str(value.get("_result_fail", "Unknown error"))
                    processed[key] = r[t.Tests.PayloadValue].fail(error_msg)
                else:
                    processed[key] = value
            else:
                processed[key] = value
        return processed

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
            # Check if value is a dict (not BaseModel)
            if isinstance(value, dict):
                items.extend(
                    self._flatten_dict(value, new_key, sep).items(),
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

            Uses r[T] directly for type-safe result creation.
            """

            @staticmethod
            def ok[T](value: T) -> r[T]:
                """Create success result using r[T] directly."""
                return r[T].ok(value)

            @staticmethod
            def fail[T](
                error: str,
                code: str | None = None,
                data: t.ConfigMap | None = None,
                expected_type: type[T] | None = None,
            ) -> r[T]:
                """Create failure result using r[T] directly."""
                _ = expected_type
                error_code = code or c.Errors.VALIDATION_ERROR
                return r[T].fail(error, error_code=error_code, error_data=data)

            @staticmethod
            def batch_ok[T](values: Sequence[T]) -> list[r[T]]:
                """Create batch of success results - DELEGATES to tt.results()."""
                return tt.results(values=list(values))

            @staticmethod
            def batch_fail[T](
                errors: Sequence[str],
                expected_type: type[T] | None = None,
            ) -> list[r[T]]:
                """Create batch of failure results - DELEGATES to tt.results()."""
                _ = expected_type
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
                """Assert success - r[T] satisfies protocol p.Result[T]."""
                return u.Tests.Result.assert_success(result)

            @staticmethod
            def assert_failure(result: r[t.Tests.PayloadValue]) -> str:
                """Assert failure - r[T] satisfies protocol p.Result[T]."""
                return u.Tests.Result.assert_failure(result)

        class Batch:
            """Batch operations - tb.Tests.Batch.*.

            DELEGATES TO: u.Tests.GenericHelpers.*, u.Tests.TestCaseHelpers.*
            """

            @staticmethod
            def scenarios[T](
                *cases: tuple[str, T],
            ) -> list[tuple[str, T]]:
                """Create parametrized test cases."""
                return list(cases)

            @staticmethod
            def from_dict[T](
                mapping: Mapping[str, T],
            ) -> list[tuple[str, T]]:
                """Convert dict to parametrized cases."""
                return list(mapping.items())

            @staticmethod
            def parametrized(
                success_values: Sequence[t.Tests.PayloadValue],
                failure_errors: Sequence[str],
            ) -> list[tuple[str, Mapping[str, t.Tests.PayloadValue]]]:
                """Create parametrized cases - DELEGATES to u.Tests.GenericHelpers."""
                cases = u.Tests.GenericHelpers.create_parametrized_cases(
                    success_values=list(success_values),
                    failure_errors=list(failure_errors),
                )
                # Convert to (test_id, data) format
                # Store result as metadata dict, not FlextResult directly
                parametrized: list[tuple[str, Mapping[str, t.Tests.PayloadValue]]] = []
                for i, (_res, is_success, value, error) in enumerate(cases):
                    test_id = f"case_{i}"
                    # Store result metadata - FlextResult stored as its components
                    data: dict[str, t.Tests.PayloadValue] = {
                        "result_is_success": is_success,
                        "result_value": value,
                        "result_error": error,
                        "is_success": is_success,
                        "value": value,
                        "error": error,
                    }
                    parametrized.append((test_id, data))
                return parametrized

            @staticmethod
            def test_cases(
                operation: str,
                descriptions: Sequence[str],
                inputs: Sequence[Mapping[str, t.Tests.PayloadValue]],
                expected: Sequence[t.Tests.PayloadValue],
            ) -> list[Mapping[str, t.Tests.PayloadValue]]:
                """Create batch test cases - DELEGATES to u.Tests.TestCaseHelpers."""
                raw_cases = u.Tests.TestCaseHelpers.create_batch_operation_test_cases(
                    operation=operation,
                    descriptions=list(descriptions),
                    input_data_list=list(inputs),
                    expected_results=list(expected),
                )
                return [dict(case) for case in raw_cases]

        class Data:
            """Data generation helpers - tb.Tests.Data.*.

            DELEGATES TO: u.Collection.*, u.Mapper.*, u.Tests.Factory.*
            """

            @staticmethod
            def typed(
                **kwargs: t.Tests.PayloadValue,
            ) -> Mapping[str, t.Tests.PayloadValue]:
                """Create typed dictionary - returns Mapping[str, t.Tests.PayloadValue]."""
                return dict(kwargs)

            @staticmethod
            def merged(
                *dicts: Mapping[str, t.Tests.PayloadValue],
            ) -> Mapping[str, t.Tests.PayloadValue]:
                """Merge dictionaries - DELEGATES to u.merge()."""
                result: MutableMapping[str, t.Tests.PayloadValue] = {}
                for d in dicts:
                    merge_result = u.merge(
                        {
                            k: FlextTestsBuilders._to_guard_input(v)
                            for k, v in result.items()
                        },
                        {
                            str(k): FlextTestsBuilders._to_guard_input(
                                FlextTestsBuilders._to_payload_value(v)
                            )
                            for k, v in d.items()
                        },
                    )
                    if merge_result.is_success:
                        result = {
                            str(k): FlextTestsBuilders._to_payload_value(v)
                            for k, v in merge_result.value.items()
                        }
                return result

            @staticmethod
            def flatten(
                nested: Mapping[str, t.Tests.PayloadValue],
                separator: str = ".",
            ) -> Mapping[str, t.Tests.PayloadValue]:
                """Flatten nested dict - uses manual implementation."""

                # Manual flatten since u.Collection.flatten may not exist
                def _flatten(
                    data: Mapping[str, t.Tests.PayloadValue],
                    parent: str = "",
                ) -> Mapping[str, t.Tests.PayloadValue]:
                    items: list[tuple[str, t.Tests.PayloadValue]] = []
                    for key, value in data.items():
                        new_key = f"{parent}{separator}{key}" if parent else key
                        if isinstance(value, Mapping):
                            value_dict: dict[str, t.Tests.PayloadValue] = {}
                            for k, v in value.items():
                                if v is None:
                                    value_dict[str(k)] = None
                                elif isinstance(
                                    v, (str, bool, int, float, list, dict, BaseModel)
                                ):
                                    value_dict[str(k)] = v
                                else:
                                    # Convert other types to string
                                    value_dict[str(k)] = str(v)
                            items.extend(_flatten(value_dict, new_key).items())
                        else:
                            items.append((new_key, value))
                    return dict(items)

                return _flatten(dict(nested))

            @staticmethod
            def transform[T, U](
                items: Sequence[T],
                func: Callable[[T], U],
            ) -> list[U]:
                """Transform items using list comprehension."""
                return [func(item) for item in items]

            @staticmethod
            def id() -> str:
                """Generate UUID - DELEGATES to u.Tests.Factory."""
                return u.Tests.Factory.generate_id()

            @staticmethod
            def short_id(length: int = 8) -> str:
                """Generate short ID - DELEGATES to u.Tests.Factory."""
                return u.Tests.Factory.generate_short_id(length)

        class Model:
            """Model creation helpers - tb.Tests.Model.*.

            DELEGATES TO: tt.model(), tt.batch(), u.Tests.DomainHelpers.*
            """

            @staticmethod
            def user(**overrides: t.Tests.PayloadValue) -> m.Tests.Factory.User:
                """Create user - DELEGATES to tt.model()."""
                result = tt.model("user", **overrides)
                if isinstance(result, m.Tests.Factory.User):
                    return result
                raise TypeError(
                    f"Expected User from tt.model('user'), got {type(result).__name__}",
                )

            @staticmethod
            def config(**overrides: t.Tests.PayloadValue) -> m.Tests.Factory.Config:
                """Create config - DELEGATES to tt.model()."""
                result = tt.model("config", **overrides)
                if isinstance(result, m.Tests.Factory.Config):
                    return result
                raise TypeError(
                    f"Expected Config from tt.model('config'), got {type(result).__name__}",
                )

            @staticmethod
            def service(**overrides: t.Tests.PayloadValue) -> m.Tests.Factory.Service:
                """Create service - DELEGATES to tt.model()."""
                result = tt.model("service", **overrides)
                if isinstance(result, m.Tests.Factory.Service):
                    return result
                raise TypeError(
                    f"Expected Service from tt.model('service'), got {type(result).__name__}",
                )

            @staticmethod
            def entity[T: m.Tests.Factory.Entity](
                entity_class: type[T],
                name: str = "",
                value: t.Tests.PayloadValue = "",
            ) -> T:
                """Create entity - DELEGATES to u.Tests.DomainHelpers."""

                def entity_factory(
                    *,
                    name: str,
                    value: t.Tests.PayloadValue,
                ) -> T:
                    return entity_class(name=name, value=value)

                return u.Tests.DomainHelpers.create_test_entity_instance(
                    name=name,
                    value=value,
                    entity_class=entity_factory,
                )

            @staticmethod
            def value_object[T: m.Tests.Factory.Value](
                value_class: type[T],
                data: str = "",
                count: int = 1,
            ) -> T:
                """Create value object - DELEGATES to u.Tests.DomainHelpers."""

                def value_factory(
                    *,
                    data: str,
                    count: int,
                ) -> T:
                    return value_class(data=data, count=count)

                return u.Tests.DomainHelpers.create_test_value_object_instance(
                    data=data,
                    count=count,
                    value_class=value_factory,
                )

            @staticmethod
            def batch_users(
                count: int = c.Tests.Factory.DEFAULT_BATCH_COUNT,
            ) -> list[m.Tests.Factory.User]:
                """Create batch users - DELEGATES to tt.batch()."""
                # tt.batch returns list of models - filter to User instances
                batch_result = tt.batch("user", count=count)
                return [
                    item
                    for item in batch_result
                    if isinstance(item, m.Tests.Factory.User)
                ]

            @staticmethod
            def batch_entities[T: m.Tests.Factory.Entity](
                entity_class: type[T],
                names: Sequence[str],
                values: Sequence[t.Tests.PayloadValue],
            ) -> list[T]:
                """Create batch entities - DELEGATES to u.Tests.DomainHelpers."""

                def entity_factory(
                    *,
                    name: str,
                    value: t.Tests.PayloadValue,
                ) -> T:
                    return entity_class(name=name, value=value)

                result: r[list[T]] = u.Tests.DomainHelpers.create_test_entities_batch(
                    names=list(names),
                    values=list(values),
                    entity_class=entity_factory,
                )
                if result.is_success and result.value is not None:
                    return result.value
                raise ValueError(result.error)

        class Operation:
            """Operation helpers - tb.Tests.Operation.*.

            DELEGATES TO: u.Tests.Factory.*, tt.op()
            """

            @staticmethod
            def simple() -> Callable[[], t.Tests.PayloadValue]:
                """Simple operation - DELEGATES to u.Tests.Factory."""
                return u.Tests.Factory.simple_operation

            @staticmethod
            def add() -> Callable[
                [t.Tests.PayloadValue, t.Tests.PayloadValue],
                t.Tests.PayloadValue,
            ]:
                """Add operation - DELEGATES to u.Tests.Factory."""
                return u.Tests.Factory.add_operation

            @staticmethod
            def format() -> Callable[[str, int], str]:
                """Format operation - DELEGATES to u.Tests.Factory."""
                return u.Tests.Factory.format_operation

            @staticmethod
            def error(message: str) -> Callable[[], t.Tests.PayloadValue]:
                """Error operation - DELEGATES to u.Tests.Factory."""
                return u.Tests.Factory.create_error_operation(message)

            @staticmethod
            def execute_service(
                overrides: Mapping[str, t.Tests.PayloadValue] | None = None,
            ) -> r[t.Tests.PayloadValue]:
                """Execute service - DELEGATES to u.Tests.Factory."""
                return u.Tests.Factory.execute_user_service(overrides or {})


# Short alias for convenient test usage
tb = FlextTestsBuilders

__all__ = ["FlextTestsBuilders", "tb"]
