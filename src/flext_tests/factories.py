"""Test data factories for FLEXT ecosystem tests.

Provides comprehensive factory pattern implementation for creating test objects,
services, and domain models. Extends FlextService for consistent architecture
and integrates with FlextModels for type-safe test data generation.

Key Features:
- Model factories for User, Config, Service, Command, Query, Entity, Value
- Result factories for FlextResult creation in tests
- Service factories for creating test service classes dynamically
- Operation factories for callable test operations
- Batch creation methods for generating multiple test objects
- Integration with FlextModels for CQRS and DDD patterns

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import builtins
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Never, TypeVar, cast

from pydantic import BaseModel as _BaseModel, PrivateAttr

from flext_core import FlextResult, r
from flext_core.typings import t as t_core
from flext_tests.base import s
from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.typings import t
from flext_tests.utilities import u

TModel = TypeVar("TModel")
TValue = TypeVar("TValue")

# Type alias for model result union - used for early return type matching
# Use BaseModel as base type since all factory models are Pydantic models

type _ModelResult = r[_BaseModel]


class FlextTestsFactories(s[t_core.GeneralValueType]):
    """Comprehensive test data factories extending FlextService.

    Provides factory methods for creating test objects, services, and domain
    models using the FlextModels foundation. Follows Railway-Oriented Programming
    with FlextResult[T] returns for error-safe test data generation.

    Architecture:
        - Extends FlextService for consistent service patterns
        - Uses FlextModels (m.Value, m.Entity) for domain models
        - Returns FlextResult[T] for operations that can fail
        - Provides both static and instance methods

    Usage:
        # Static factory methods (most common)
        user = FlextTestsFactories.create_user(name="John")
        config = FlextTestsFactories.create_config(debug=True)

        # Instance for service-based operations
        factory = FlextTestsFactories()
        result = factory.execute()  # Returns FlextResult[TestResultValue]

        # Result factories for test assertions
        success = FlextTestsFactories.Result.ok("value")
        failure = FlextTestsFactories.Result.fail("error")
    """

    # ==========================================================================
    # GENERALIST FACTORY METHODS - Preferred API (use tt.method())
    # ==========================================================================
    # Models: m.Tests.Factory namespace - NO local aliases
    # Types: t.Tests.Factory namespace (from flext_tests.typings)

    @classmethod
    def model(
        cls,
        kind: t.Tests.Factory.ModelKind = "user",
        # All parameters via kwargs - will be validated by ModelFactoryParams
        **kwargs: t.Tests.TestResultValue,
    ) -> (
        _BaseModel
        | builtins.list[_BaseModel]
        | dict[str, _BaseModel]
        | r[_BaseModel]
        | r[list[_BaseModel]]
        | r[dict[str, _BaseModel]]
    ):
        """Unified model factory - creates any model type with full customization.

        This is the preferred way to create test models. Use tt.model() instead of
        individual create_* methods.

        Args:
            kind: Model type to create ('user', 'config', 'service', 'entity', 'value', 'command', 'query', 'event')
            count: Number of instances to create (returns list if count > 1)
            as_dict: Return as dict with ID keys
            as_result: Wrap in FlextResult
            as_mapping: Map to custom keys (Mapping[str, str])
            factory: Custom factory callable
            transform: Post-transform function
            validate: Validation predicate
            model_id: Identifier (auto-generated if not provided)
            name: Name field (varies by model type)
            email: Email for user models
            active: Active status for user models
            service_type: Service type for config/service models
            environment: Environment for config models
            debug: Debug flag for config models
            log_level: Log level for config models
            timeout: Timeout for config models
            max_retries: Max retries for config models
            status: Status for service models
            value: Value for entity models
            data: Data for value object models
            value_count: Count for value object models
            **overrides: Override any field directly

        Returns:
            Model instance, list of models, dict of models, or FlextResult wrapping any of these

        Examples:
            # Create user
            user = tt.model("user", name="John", email="john@example.com")

            # Create batch of users
            users = tt.model("user", count=5)

            # Create user as dict
            users_dict = tt.model("user", count=3, as_dict=True)

            # Create user wrapped in result
            user_result = tt.model("user", as_result=True)

            # Create with custom factory
            user = tt.model("user", factory=lambda: m.Tests.Factory.User(id="1", name="Test", email="test@example.com"))

            # Create with transform
            user = tt.model("user", transform=lambda u: u.model_copy(update={"name": "Modified"}))

            # Create with validation
            user = tt.model("user", validate=lambda u: u.active)

        """
        # Validate and convert kwargs to ModelFactoryParams using FlextUtilities
        params_result = u.Model.from_kwargs(
            m.Tests.Factory.ModelFactoryParams,
            kind=kind,
            **kwargs,
        )
        if params_result.is_failure:
            # Return validation error as FlextResult - cast to match return type
            return cast(
                "_ModelResult",
                r[t.Tests.Factory.FactoryModel].fail(
                    f"Invalid parameters: {params_result.error}",
                ),
            )
        params = params_result.value

        # Helper to create a single model instance
        def _create_single() -> (
            m.Tests.Factory.User
            | m.Tests.Factory.Config
            | m.Tests.Factory.Service
            | m.Tests.Factory.Entity
            | m.Tests.Factory.ValueObject
        ):
            if params.factory:
                factory_result = params.factory()
                # Cast to expected model type (factory returns object, but we know it's a model)
                return cast(
                    "m.Tests.Factory.User | m.Tests.Factory.Config | m.Tests.Factory.Service | m.Tests.Factory.Entity | m.Tests.Factory.ValueObject",
                    factory_result,
                )

            if params.kind == "user":
                user_data: t.Types.ConfigurationDict = {
                    "id": params.model_id or u.Tests.Factory.generate_id(),
                    "name": params.name or c.Tests.Factory.DEFAULT_USER_NAME,
                    "email": params.email
                    or c.Tests.Factory.user_email(u.Tests.Factory.generate_short_id(8)),
                    "active": (
                        params.active
                        if params.active is not None
                        else c.Tests.Factory.DEFAULT_USER_ACTIVE
                    ),
                }
                if params.overrides:
                    # Convert overrides to ConfigurationDict for merge
                    user_overrides: t.Types.ConfigurationDict = dict(params.overrides)
                    merge_result = u.merge(user_data, user_overrides, strategy="deep")
                    if merge_result.is_success:
                        user_data = merge_result.value
                return m.Tests.Factory.User.model_validate(user_data)

            if params.kind == "config":
                config_data: t.Types.ConfigurationDict = {
                    "service_type": params.service_type
                    or c.Tests.Factory.DEFAULT_SERVICE_TYPE,
                    "environment": params.environment
                    or c.Tests.Factory.DEFAULT_ENVIRONMENT,
                    "debug": params.debug
                    if params.debug is not None
                    else c.Tests.Factory.DEFAULT_DEBUG,
                    "log_level": params.log_level or c.Tests.Factory.DEFAULT_LOG_LEVEL,
                    "timeout": params.timeout
                    if params.timeout is not None
                    else c.Tests.Factory.DEFAULT_TIMEOUT,
                    "max_retries": params.max_retries
                    if params.max_retries is not None
                    else c.Tests.Factory.DEFAULT_MAX_RETRIES,
                }
                if params.overrides:
                    # Convert overrides to ConfigurationDict for merge
                    config_overrides: t.Types.ConfigurationDict = dict(params.overrides)
                    merge_result = u.merge(
                        config_data,
                        config_overrides,
                        strategy="deep",
                    )
                    if merge_result.is_success:
                        config_data = merge_result.value
                # Create Config model - use m.Tests.Factory.Config which is a real Pydantic model
                return m.Tests.Factory.Config.model_validate(config_data)

            if params.kind == "service":
                service_type_str = params.service_type or "api"
                svc_data: dict[str, t.Tests.TestResultValue] = {
                    "id": params.model_id or u.generate("uuid"),
                    "type": service_type_str,
                    "name": params.name
                    or c.Tests.Factory.service_name(service_type_str),
                    "status": params.status or "active",
                }
                # Convert overrides to compatible dict
                if params.overrides:
                    # Type narrowing: params.overrides is Mapping[str, GeneralValueType]
                    overrides_mapping = cast(
                        "Mapping[str, t.Tests.TestResultValue]", params.overrides
                    )
                    overrides_dict: dict[str, t.Tests.TestResultValue] = dict(
                        overrides_mapping
                    )
                    svc_data.update(overrides_dict)
                # Create Service model using the proper test Service model
                return m.Tests.Factory.Service.model_validate(svc_data)

            if params.kind == "entity":
                # Use DomainHelpers for entity creation
                return u.Tests.DomainHelpers.create_test_entity_instance(
                    name=params.name or c.Tests.Factory.DEFAULT_ENTITY_NAME,
                    value=params.value,
                    entity_class=m.Tests.Factory.Entity,
                )

            # params.kind == "value"
            # Use DomainHelpers for value object creation with proper ValueObject model
            value_data = params.data or "default_value"
            value_count = params.value_count or 1
            return u.Tests.DomainHelpers.create_test_value_object_instance(
                data=value_data,
                count=value_count,
                value_class=m.Tests.Factory.ValueObject,
            )

        # Create single instance
        instance = _create_single()

        # Apply transform if provided
        if params.transform:
            transformed = params.transform(instance)
            # Cast to expected model type
            instance = cast(
                "m.Tests.Factory.User | m.Tests.Factory.Config | m.Tests.Factory.Service | m.Tests.Factory.Entity | m.Tests.Factory.ValueObject",
                transformed,
            )

        # Apply validation if provided
        if params.validate_fn and not params.validate_fn(instance):
            return cast(
                "_ModelResult",
                r[t.Tests.Factory.FactoryModel].fail(c.Tests.Factory.ERROR_VALIDATION),
            )

        # Handle count > 1
        if params.count > 1:
            instances: builtins.list[
                m.Tests.Factory.User
                | m.Tests.Factory.Config
                | m.Tests.Factory.Service
                | m.Tests.Factory.Entity
                | m.Tests.Factory.ValueObject
            ] = [instance]
            for _ in range(params.count - 1):
                new_instance = _create_single()
                if params.transform:
                    transformed = params.transform(new_instance)
                    # Cast to expected model type
                    new_instance = cast(
                        "m.Tests.Factory.User | m.Tests.Factory.Config | m.Tests.Factory.Service | m.Tests.Factory.Entity | m.Tests.Factory.ValueObject",
                        transformed,
                    )
                if params.validate_fn and not params.validate_fn(new_instance):
                    return r[list[_BaseModel]].fail(c.Tests.Factory.ERROR_VALIDATION)
                instances.append(new_instance)

            # Handle as_dict
            if params.as_dict:
                result_dict: dict[str, _BaseModel] = {}
                for inst in instances:
                    # Try to get ID from common attributes using getattr
                    inst_id = (
                        getattr(inst, "id", None)
                        or getattr(inst, "model_id", None)
                        or u.Tests.Factory.generate_id()
                    )
                    result_dict[str(inst_id)] = inst
                if params.as_result:
                    dict_result: r[dict[str, _BaseModel]] = r[dict[str, _BaseModel]].ok(
                        result_dict
                    )
                    return dict_result
                return result_dict

            # Handle as_mapping
            if params.as_mapping:
                mapped_result_dict: dict[str, _BaseModel] = {}
                for i, inst in enumerate(instances):
                    key = params.as_mapping.get(str(i), str(i))
                    mapped_result_dict[key] = inst
                if params.as_result:
                    mapping_result: r[dict[str, _BaseModel]] = r[
                        dict[str, _BaseModel]
                    ].ok(mapped_result_dict)
                    return mapping_result
                return mapped_result_dict

            # Return list
            typed_instances: list[_BaseModel] = [
                cast("_BaseModel", inst) for inst in instances
            ]
            if params.as_result:
                return r[list[_BaseModel]].ok(typed_instances)
            return typed_instances

        # Single instance - handle as_dict
        typed_instance = instance
        if params.as_dict:
            inst_id = (
                getattr(typed_instance, "id", None)
                or getattr(typed_instance, "model_id", None)
                or u.Tests.Factory.generate_id()
            )
            single_result_dict: dict[str, _BaseModel] = {
                str(inst_id): cast("_BaseModel", typed_instance)
            }
            if params.as_result:
                single_dict_result: r[dict[str, _BaseModel]] = r[
                    dict[str, _BaseModel]
                ].ok(single_result_dict)
                return single_dict_result
            return single_result_dict

        # Handle as_mapping for single instance
        if params.as_mapping:
            key = params.as_mapping.get("0", "0")
            single_mapped_dict: dict[str, _BaseModel] = {
                key: cast("_BaseModel", typed_instance)
            }
            if params.as_result:
                single_mapping_result: r[dict[str, _BaseModel]] = r[
                    dict[str, _BaseModel]
                ].ok(single_mapped_dict)
                return single_mapping_result
            return single_mapped_dict

        # Return single instance
        if params.as_result:
            # Cast to match return type
            return cast(
                "_ModelResult",
                r[t.Tests.Factory.FactoryModel].ok(typed_instance),
            )
        return typed_instance

    @classmethod
    def res[TValue](
        cls,
        kind: t.Tests.Factory.ResultKind = "ok",
        value: TValue | None = None,
        # All parameters via kwargs - will be validated by ResultFactoryParams
        **kwargs: t.Tests.TestResultValue,
    ) -> r[TValue] | builtins.list[r[TValue]]:
        """Unified result factory - creates FlextResult with full customization.

        This is the preferred way to create test results. Use tt.res() instead of
        Result.ok(), Result.fail(), etc.

        Args:
            kind: Result type ('ok', 'fail', 'from_value')
            value: Value for success (required for 'ok')
            count: Number of results to create (returns list if count > 1)
            values: Explicit value list for batch creation
            errors: Error messages for failure results
            mix_pattern: Success/failure pattern (True=success, False=failure)
            error: Error message for failure results
            error_code: Optional error code for failure results
            error_on_none: Error message when value is None (for 'from_value')
            transform: Transform function for success values

        Returns:
            FlextResult instance or list of FlextResult instances

        Examples:
            # Success result
            ok = tt.res("ok", value="success_data")

            # Failure result
            fail = tt.res("fail", error="Something went wrong", error_code="ERR001")

            # From optional value (fails if None)
            maybe = tt.res("from_value", value=some_optional, error_on_none="Required!")

            # Batch results
            results = tt.res("ok", values=[1, 2, 3])

            # Mixed pattern
            results = tt.res("ok", values=[1, 2], errors=["e1", "e2"], mix_pattern=[True, False, True, False])

        """
        # Validate and convert kwargs to ResultFactoryParams using FlextUtilities
        # Pass kind and value explicitly, then merge with kwargs
        params_result = u.Model.from_kwargs(
            m.Tests.Factory.ResultFactoryParams,
            kind=kind,
            value=value,
            **kwargs,
        )
        if params_result.is_failure:
            # Return validation error as FlextResult
            return r[TValue].fail(f"Invalid parameters: {params_result.error}")
        params = params_result.value

        # Handle batch creation with mix_pattern
        if params.mix_pattern is not None and (
            params.values is not None or params.errors is not None
        ):
            result_list: builtins.list[r[TValue]] = []
            val_idx = 0
            err_idx = 0
            for is_success in params.mix_pattern:
                if is_success:
                    if params.values and val_idx < len(params.values):
                        val = cast("TValue", params.values[val_idx])
                        if params.transform:
                            val = cast("TValue", params.transform(val))
                        result_list.append(r[TValue].ok(val))
                        val_idx += 1
                    elif params.value is not None:
                        val = cast("TValue", params.value)
                        if params.transform:
                            val = cast("TValue", params.transform(val))
                        result_list.append(r[TValue].ok(val))
                elif params.errors and err_idx < len(params.errors):
                    result_list.append(
                        r[TValue].fail(
                            params.errors[err_idx],
                            error_code=params.error_code,
                        ),
                    )
                    err_idx += 1
                else:
                    result_list.append(
                        r[TValue].fail(params.error, error_code=params.error_code),
                    )
            return result_list

        # Handle batch creation with explicit values/errors
        if params.values is not None or params.errors is not None or params.count > 1:
            result_list = []
            if params.values:
                for raw_val in params.values:
                    # raw_val is GeneralValueType, transform if needed then cast to TValue
                    transformed_val: TValue = (
                        cast("TValue", params.transform(raw_val))
                        if params.transform
                        else cast("TValue", raw_val)
                    )
                    result_list.append(r[TValue].ok(transformed_val))
            if params.errors:
                for err in params.errors:
                    result_list.append(
                        r[TValue].fail(err, error_code=params.error_code),
                    )
            if params.count > 1 and not params.values and not params.errors:
                # Create multiple results from single value
                for _ in range(params.count):
                    if params.value is None:
                        if params.error_on_none:
                            result_list.append(r[TValue].fail(params.error_on_none))
                        else:
                            # None is allowed when TValue includes None
                            result_list.append(
                                r[TValue].ok(cast("TValue", params.value)),
                            )
                    else:
                        val = cast("TValue", params.value)
                        if params.transform:
                            val = cast("TValue", params.transform(val))
                        result_list.append(r[TValue].ok(val))
            if result_list:
                return result_list
            # Empty list case
            if params.value is not None:
                return [r[TValue].ok(cast("TValue", params.value))]
            return [r[TValue].fail(params.error_on_none or params.error)]

        # Single result creation
        if params.kind == "ok":
            if params.value is None:
                # Return empty success for None value (TValue can be None)
                return r[TValue].ok(cast("TValue", params.value))
            raw_value = cast("TValue", params.value)
            transformed_value = (
                cast("TValue", params.transform(raw_value))
                if params.transform
                else raw_value
            )
            return r[TValue].ok(transformed_value)

        if params.kind == "fail":
            return r[TValue].fail(params.error, error_code=params.error_code)

        # params.kind == "from_value"
        if params.value is None:
            return r[TValue].fail(
                params.error_on_none or c.Tests.Factory.ERROR_VALUE_NONE,
            )
        raw_value = cast("TValue", params.value)
        transformed_value = (
            cast("TValue", params.transform(raw_value))
            if params.transform
            else raw_value
        )
        return r[TValue].ok(transformed_value)

    @classmethod
    def op(
        cls,
        kind: t.Tests.Factory.OpKind = "simple",
        *,
        error_message: str = c.Tests.Factory.ERROR_DEFAULT,
        result_value: t.Tests.TestResultValue = c.Tests.Factory.SUCCESS_MESSAGE,
    ) -> Callable[..., t.Tests.TestResultValue | r[t.Tests.TestResultValue]]:
        """Unified operation factory - creates callable test operations.

        This is the preferred way to create test operations. Use tt.op() instead of
        Operations.simple(), Operations.error(), etc.

        Args:
            kind: Operation type
                - 'simple': Returns "success" string
                - 'add': Adds two values (numeric or string concat)
                - 'format': Formats "name: value" string
                - 'error': Raises ValueError
                - 'type_error': Raises TypeError
                - 'result_ok': Returns FlextResult.ok()
                - 'result_fail': Returns FlextResult.fail()
            error_message: Error message for error operations
            result_value: Value for result_ok operation

        Returns:
            Callable operation

        Examples:
            # Simple operation
            simple = tt.op("simple")
            assert simple() == "success"

            # Error operation
            error_op = tt.op("error", error_message="Custom error")
            # error_op() raises ValueError("Custom error")

            # Result operations
            ok_op = tt.op("result_ok", result_value=42)
            fail_op = tt.op("result_fail", error_message="Failed!")

        """
        if kind == "simple":

            def simple_op() -> str:
                return c.Tests.Factory.SUCCESS_MESSAGE

            return simple_op

        if kind == "add":

            def add_op(
                a: t.Tests.TestResultValue,
                b: t.Tests.TestResultValue,
            ) -> t.Tests.TestResultValue:
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return a + b
                return str(a) + str(b)

            return add_op

        if kind == "format":

            def format_op(name: str, value: int = 10) -> str:
                return f"{name}: {value}"

            return format_op

        if kind == "error":

            def error_op() -> Never:
                raise ValueError(error_message)

            return error_op

        if kind == "type_error":

            def type_error_op() -> Never:
                raise TypeError(error_message)

            return type_error_op

        if kind == "result_ok":

            def result_ok_op() -> r[t.Tests.TestResultValue]:
                return r[t.Tests.TestResultValue].ok(result_value)

            return result_ok_op

        # kind == "result_fail"
        def result_fail_op() -> r[t.Tests.TestResultValue]:
            return r[t.Tests.TestResultValue].fail(error_message)

        return result_fail_op

    @classmethod
    def batch[TModel](
        cls,
        kind: t.Tests.Factory.ModelKind = "user",
        count: int = c.Tests.Factory.DEFAULT_BATCH_COUNT,
        *,
        # Batch-specific options
        names: Sequence[str] | None = None,
        environments: Sequence[str] | None = None,
        service_types: Sequence[str] | None = None,
        # Common customization
        **common_overrides: t.Tests.TestResultValue,
    ) -> (
        list[m.Tests.Factory.User]
        | builtins.list[m.Tests.Factory.Config]
        | builtins.list[m.Tests.Factory.Service]
    ):
        """Unified batch factory - creates multiple model instances.

        This is the preferred way to create batches of test models.
        Use tt.batch() instead of batch_users(), Batch.configs(), etc.

        Args:
            kind: Model type to create ('user', 'config', 'service')
            count: Number of instances to create
            names: Optional list of names to cycle through
            environments: Optional list of environments (for configs)
            service_types: Optional list of service types (for services)
            **common_overrides: Fields applied to all instances

        Returns:
            List of model instances

        Examples:
            # Batch of users
            users = tt.batch("user", count=10)

            # Batch of configs with environments
            configs = tt.batch("config", count=3, environments=["dev", "prod"])

            # Batch of services with common override
            services = tt.batch("service", count=5, status="pending")

        """
        # Note: common_overrides ignored for type safety - use named params instead
        _ = common_overrides  # Explicitly ignore to avoid unused warning

        if kind == "user":
            result_users: builtins.list[m.Tests.Factory.User] = []
            for i in range(count):
                name = names[i % len(names)] if names else f"User {i}"
                result_users.append(
                    cast(
                        "m.Tests.Factory.User",
                        cls.model("user", name=name, email=f"user{i}@example.com"),
                    ),
                )
            return result_users

        if kind == "config":
            envs = (
                list(environments)
                if environments
                else list(c.Tests.Factory.DEFAULT_BATCH_ENVIRONMENTS)
            )
            return [
                cast(
                    "m.Tests.Factory.Config",
                    cls.model("config", environment=envs[i % len(envs)]),
                )
                for i in range(count)
            ]

        # kind == "service"
        types = (
            list(service_types)
            if service_types
            else list(c.Tests.Factory.DEFAULT_BATCH_SERVICE_TYPES)
        )
        return [
            cast(
                "m.Tests.Factory.Service",
                cls.model("service", service_type=types[i % len(types)]),
            )
            for i in range(count)
        ]

    @classmethod
    def results[TValue](
        cls,
        values: Sequence[TValue],
        *,
        errors: Sequence[str] | None = None,
        mix_pattern: Sequence[bool] | None = None,
    ) -> builtins.list[r[TValue]]:
        """Create batch of FlextResult instances from values and errors.

        Args:
            values: Values for success results
            errors: Error messages for failure results (appended after successes)
            mix_pattern: If provided, interleaves success/failure based on pattern
                         True = use value, False = use error

        Returns:
            List of FlextResult instances

        Examples:
            # All successes
            results = tt.results([1, 2, 3])

            # Successes + failures
            results = tt.results([1, 2], errors=["err1", "err2"])

            # Mixed pattern: success, fail, success, fail
            mix = [True, False, True, False]
            results = tt.results([1, 2], errors=["e1", "e2"], mix_pattern=mix)

        """
        if mix_pattern and errors:
            result_list: builtins.list[r[TValue]] = []
            val_idx = 0
            err_idx = 0
            for is_success in mix_pattern:
                if is_success and val_idx < len(values):
                    result_list.append(r[TValue].ok(values[val_idx]))
                    val_idx += 1
                elif not is_success and err_idx < len(errors):
                    result_list.append(r[TValue].fail(errors[err_idx]))
                    err_idx += 1
            return result_list

        # Default: all successes first, then failures
        result_list = [r[TValue].ok(v) for v in values]
        if errors:
            result_list.extend([r[TValue].fail(e) for e in errors])
        return result_list

    @classmethod
    def list[T](
        cls,
        source: (Sequence[T] | Callable[[], T] | t.Tests.Factory.ModelKind) = "user",
        # All parameters via kwargs - will be validated by ListFactoryParams
        **kwargs: t.Tests.TestResultValue,
    ) -> builtins.list[T] | r[builtins.list[T]]:
        """Create typed list from source.

        This is the preferred way to create lists of test data.
        Use tt.list() instead of manual list comprehensions.

        Args:
            source: Source for list items:
                - Sequence[T]: Use items directly
                - Callable[[], T]: Factory function to call repeatedly
                - ModelKind: Create models of this kind (delegates to tt.model())
            count: Number of items to create (if source is callable or ModelKind)
            as_result: Wrap result in FlextResult
            unique: Ensure all items are unique (if applicable)
            transform: Transform function applied to each item
            filter_: Filter predicate to exclude items

        Returns:
            List of items or FlextResult wrapping list

        Examples:
            # List from model kind
            users = tt.list("user", count=3)
            assert len(users) == 3
            assert all(isinstance(u, m.Tests.Factory.User) for u in users)

            # List from callable
            numbers = tt.list(lambda: 42, count=5)
            assert numbers == [42, 42, 42, 42, 42]

            # List from sequence with transform
            doubled = tt.list([1, 2, 3], transform=lambda x: x * 2)
            assert doubled == [2, 4, 6]

            # List with filter
            evens = tt.list([1, 2, 3, 4, 5], filter_=lambda x: x % 2 == 0)
            assert evens == [2, 4]

            # List wrapped in result
            result = tt.list("user", count=3, as_result=True)
            assert result.is_success
            assert len(result.value) == 3

        """
        # Validate and convert kwargs to ListFactoryParams using FlextUtilities
        # Pass source explicitly, then merge with kwargs
        params_result = u.Model.from_kwargs(
            m.Tests.Factory.ListFactoryParams,
            source=source,
            **kwargs,
        )
        if params_result.is_failure:
            # Return validation error as FlextResult
            return r[builtins.list[T]].fail(
                f"Invalid parameters: {params_result.error}"
            )
        params = params_result.value

        items: builtins.list[T] = []

        # Handle different source types
        if u.is_type(params.source, "str"):
            # ModelKind - delegate to tt.model()
            for _ in range(params.count):
                model_result = cls.model(
                    cast("t.Tests.Factory.ModelKind", params.source),
                    count=1,
                )
                # Extract item from model result (can be model, list, or FlextResult)
                raw_item: object
                if isinstance(model_result, list):
                    raw_item = model_result[0]
                elif isinstance(model_result, FlextResult):
                    # isinstance() narrows to FlextResult[...], has is_success and unwrap()
                    if model_result.is_success:
                        raw_item = model_result.unwrap()
                    else:
                        continue  # Skip failed results
                else:
                    raw_item = model_result
                # Cast to T (model factories return specific model types)
                item = cast("T", raw_item)
                if params.transform:
                    item = cast("T", params.transform(item))
                if params.filter_ is None or params.filter_(item):
                    items.append(item)

        elif callable(params.source):
            # Callable factory - cast after callable() check for type narrowing
            source_callable: Callable[[], object] = params.source
            for _ in range(params.count):
                raw_item = source_callable()
                item = cast("T", raw_item)
                if params.transform:
                    item = cast("T", params.transform(item))
                if params.filter_ is None or params.filter_(item):
                    items.append(item)

        elif u.is_type(params.source, "sequence_not_str"):
            # Sequence - use items directly (exclude strings)
            source_seq = cast("Sequence[T]", params.source)
            for source_item in source_seq:
                # Use separate variable for transformed item to maintain type safety
                final_item: T
                if params.transform:
                    final_item = cast("T", params.transform(source_item))
                else:
                    final_item = source_item
                if params.filter_ is None or params.filter_(final_item):
                    items.append(final_item)

        # Ensure uniqueness if requested
        if params.unique and items:
            seen: set[object] = set()
            unique_items: builtins.list[T] = []
            for item in items:
                item_hash = (
                    hash(item)
                    if hasattr(item, "__hash__") and item.__hash__ is not None
                    else id(item)
                )
                if item_hash not in seen:
                    seen.add(item_hash)
                    unique_items.append(item)
            items = unique_items

        if params.as_result:
            return r[builtins.list[T]].ok(items)
        return items

    @classmethod
    def dict_factory[K, V](
        cls,
        source: (
            Mapping[K, V] | Callable[[], tuple[K, V]] | t.Tests.Factory.ModelKind
        ) = "user",
        # All parameters via kwargs - will be validated by DictFactoryParams
        **kwargs: t.Tests.TestResultValue,
    ) -> dict[K, V] | r[dict[K, V]]:
        """Create typed dict from source.

        This is the preferred way to create dicts of test data.
        Use tt.dict_factory() instead of manual dict comprehensions.

        Args:
            source: Source for dict items:
                - Mapping[K, V]: Use mapping directly
                - Callable[[], tuple[K, V]]: Factory function returning (key, value)
                - ModelKind: Create models as values with auto-generated keys
            count: Number of items to create (if source is callable or ModelKind)
            key_factory: Factory function for keys (takes index, returns K)
            value_factory: Factory function for values (takes key, returns V)
            as_result: Wrap result in FlextResult
            merge_with: Additional mapping to merge into result

        Returns:
            Dict of items or FlextResult wrapping dict

        Examples:
            # Dict from model kind
            users = tt.dict_factory("user", count=3)
            assert len(users) == 3
            assert all(isinstance(u, m.Tests.Factory.User) for u in users.values())

            # Dict from callable
            pairs = tt.dict_factory(lambda: (f"key_{i}", i), count=3)
            # Note: callable doesn't receive index, use key_factory instead

            # Dict with key/value factories
            data = tt.dict_factory(
                "user",
                count=3,
                key_factory=lambda i: f"user_{i}",
                value_factory=lambda k: cls.model("user", name=k),
            )

            # Dict from mapping
            existing = {"a": 1, "b": 2}
            merged = tt.dict_factory(existing, merge_with={"c": 3})
            assert merged == {"a": 1, "b": 2, "c": 3}

            # Dict wrapped in result
            result = tt.dict_factory("user", count=3, as_result=True)
            assert result.is_success
            assert len(result.value) == 3

        """
        # Validate and convert kwargs to DictFactoryParams using FlextUtilities
        # Pass source explicitly, then merge with kwargs
        params_result = u.Model.from_kwargs(
            m.Tests.Factory.DictFactoryParams,
            source=source,
            **kwargs,
        )
        if params_result.is_failure:
            # Return validation error as FlextResult
            return r[dict[K, V]].fail(f"Invalid parameters: {params_result.error}")
        params = params_result.value

        result_dict: dict[K, V] = {}

        # Handle different source types
        if u.is_type(params.source, "str"):
            # ModelKind - create models with auto keys
            for i in range(params.count):
                key: K = cast(
                    "K",
                    params.key_factory(i) if params.key_factory else f"item_{i}",
                )

                model_result = cls.model(
                    cast("t.Tests.Factory.ModelKind", params.source),
                    count=1,
                )
                # Extract value from model result
                raw_value: object
                if isinstance(model_result, list):
                    raw_value = model_result[0]
                elif isinstance(model_result, FlextResult):
                    # isinstance() narrows to FlextResult[...], has is_success and unwrap()
                    if model_result.is_success:
                        raw_value = model_result.unwrap()
                    else:
                        continue  # Skip failed results
                else:
                    raw_value = model_result
                # Cast to V (model factories return specific model types)
                value = cast("V", raw_value)

                if params.value_factory:
                    value = cast("V", params.value_factory(key))

                result_dict[key] = value

        elif callable(params.source):
            # Callable factory returning (key, value) tuples
            source_callable = cast("Callable[[], tuple[K, V]]", params.source)
            for i in range(params.count):
                key, value = source_callable()
                if params.key_factory:
                    key = cast("K", params.key_factory(i))
                if params.value_factory:
                    value = cast("V", params.value_factory(key))
                result_dict[key] = value

        elif u.is_type(params.source, "mapping") and not u.is_type(
            params.source,
            "str",
        ):
            # Mapping - use directly (exclude strings)
            source_mapping = cast("Mapping[K, V]", params.source)
            result_dict.update(source_mapping)

        # Merge with additional mapping
        if params.merge_with:
            merge_mapping = cast("Mapping[K, V]", params.merge_with)
            result_dict.update(merge_mapping)

        if params.as_result:
            return r[dict[K, V]].ok(result_dict)
        return result_dict

    @classmethod
    def generic[T](
        cls,
        type_: type[T],
        # All parameters via kwargs - will be validated by GenericFactoryParams
        **kwargs: t.Tests.TestResultValue,
    ) -> T | builtins.list[T] | r[T] | r[builtins.list[T]]:
        """Create instance(s) of any type with full type safety.

        This is the preferred way to instantiate generic types.
        Use tt.generic() instead of manual instantiation.

        Args:
            type_: Type class to instantiate
            args: Positional arguments for constructor
            kwargs: Keyword arguments for constructor
            count: Number of instances to create (returns list if count > 1)
            as_result: Wrap result in FlextResult
            validate: Validation predicate (must return True for success)

        Returns:
            Instance, list of instances, or FlextResult wrapping any of these

        Examples:
            # Simple instantiation
            obj = tt.generic(SomeClass, kwargs={"name": "test"})
            assert isinstance(obj, SomeClass)

            # With positional args
            obj = tt.generic(SomeClass, args=[1, 2, 3], kwargs={"name": "test"})

            # Batch creation
            objs = tt.generic(SomeClass, kwargs={"name": "test"}, count=5)
            assert len(objs) == 5

            # With validation
            obj = tt.generic(
                SomeClass,
                kwargs={"age": 25},
                validate=lambda o: o.age >= 18,
            )

            # Wrapped in result
            result = tt.generic(SomeClass, kwargs={"name": "test"}, as_result=True)
            assert result.is_success
            assert isinstance(result.value, SomeClass)

        """
        # Validate and convert kwargs to GenericFactoryParams using FlextUtilities
        # Pass type_ explicitly, then merge with kwargs
        params_result = u.Model.from_kwargs(
            m.Tests.Factory.GenericFactoryParams,
            type_=type_,
            **kwargs,
        )
        if params_result.is_failure:
            # Return validation error as FlextResult
            return r[T].fail(f"Invalid parameters: {params_result.error}")
        params = params_result.value

        args = params.args or ()
        kwargs_dict = params.kwargs or {}

        def _create_instance() -> T:
            # Cast type_ from object to type[T] (validated by model_validator)
            type_cls = cast("type[T]", params.type_)
            instance = type_cls(*args, **kwargs_dict)
            if params.validate_fn and not params.validate_fn(instance):
                type_name = getattr(type_cls, "__name__", "Unknown")
                raise ValueError(f"Validation failed for {type_name}")
            return instance

        if params.count > 1:
            instances: builtins.list[T] = []
            for _ in range(params.count):
                try:
                    instance = _create_instance()
                    instances.append(instance)
                except Exception as e:
                    if params.as_result:
                        return r[builtins.list[T]].fail(
                            f"Failed to create instance: {e}"
                        )
                    raise

            if params.as_result:
                return r[builtins.list[T]].ok(instances)
            return instances

        # Single instance
        try:
            instance = _create_instance()
            if params.as_result:
                return r[T].ok(instance)
            return instance
        except Exception as e:
            if params.as_result:
                return r[T].fail(f"Failed to create instance: {e}")
            raise

    @classmethod
    def svc(
        cls,
        kind: str = "test",
        *,
        _with_validation: bool = False,
        **overrides: t.Tests.TestResultValue,
    ) -> type:
        """Create dynamic test service class.

        This is the preferred way to create test service classes.
        Use tt.svc() instead of create_test_service().

        Args:
            kind: Service kind ('test', 'user', 'complex')
            _with_validation: Reserved for future use (complex validation)
            **overrides: Additional attributes for the service

        Returns:
            Test service class (not instance)

        Examples:
            # Simple test service
            TestSvc = tt.svc()
            svc = TestSvc()
            result = svc.execute()

            # Complex service with validation
            ComplexSvc = tt.svc("complex")
            svc = ComplexSvc(name="test", amount=100)

        """
        return cls.create_test_service(kind, **overrides)

    # ==========================================================================
    # DEPRECATED NESTED FACTORY CLASSES - Use tt.model(), tt.res(), tt.op(), tt.batch()
    # ==========================================================================

    class Result:
        """FlextResult factory methods for test assertions.

        .. deprecated::
            Use tt.res() instead. This class will be removed in a future version.

        Provides simplified creation of success/failure results for testing.
        """

        @staticmethod
        def ok[TValue](value: TValue) -> r[TValue]:
            """Create success result with value.

            .. deprecated::
                Use tt.res("ok", value=value) instead.

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_RESULT_OK,
                DeprecationWarning,
                stacklevel=2,
            )
            return r[TValue].ok(value)

        @staticmethod
        def fail(error: str, error_code: str | None = None) -> r[object]:
            """Create failure result with error.

            .. deprecated::
                Use tt.res("fail", error=error, error_code=error_code) instead.

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_RESULT_FAIL,
                DeprecationWarning,
                stacklevel=2,
            )
            return r[object].fail(error, error_code=error_code)

        @staticmethod
        def from_value[TValue](
            value: TValue | None,
            error_on_none: str = c.Tests.Factory.ERROR_VALUE_NONE,
        ) -> r[TValue]:
            """Create result from optional value.

            .. deprecated::
                Use tt.res("from_value", value=value, error_on_none=msg) instead.

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_RESULT_FROM_VALUE,
                DeprecationWarning,
                stacklevel=2,
            )
            if value is None:
                return r[TValue].fail(error_on_none)
            return r[TValue].ok(value)

    class Models:
        """Model factory methods for domain objects.

        .. deprecated::
            Use tt.model() instead. This class will be removed in a future version.

        Creates instances of common domain models with sensible defaults.
        """

        @staticmethod
        def user(
            user_id: str | None = None,
            name: str | None = None,
            email: str | None = None,
            **overrides: t.Tests.TestResultValue,
        ) -> m.Tests.Factory.User:
            """Create test user model.

            .. deprecated::
                Use tt.model("user", id=user_id, name=name, email=email) instead.

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_MODELS_USER,
                DeprecationWarning,
                stacklevel=2,
            )
            return FlextTestsFactories.create_user(user_id, name, email, **overrides)

        @staticmethod
        def config(
            service_type: str = "api",
            environment: str = "test",
            **overrides: t.Tests.TestResultValue,
        ) -> m.Tests.Factory.Config:
            """Create test configuration model.

            .. deprecated::
                Use tt.model("config", service_type=..., environment=...) instead.

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_MODELS_CONFIG,
                DeprecationWarning,
                stacklevel=2,
            )
            return FlextTestsFactories.create_config(
                service_type,
                environment,
                **overrides,
            )

        @staticmethod
        def service(
            service_type: str = "api",
            service_id: str | None = None,
            **overrides: t.Tests.TestResultValue,
        ) -> m.Tests.Factory.Service:
            """Create test service model.

            .. deprecated::
                Use tt.model("service", service_type=..., id=...) instead.

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_MODELS_SERVICE,
                DeprecationWarning,
                stacklevel=2,
            )
            return FlextTestsFactories.create_service(
                service_type,
                service_id,
                **overrides,
            )

        @staticmethod
        def entity(
            name: str = c.Tests.Factory.DEFAULT_ENTITY_NAME,
            value: t.GeneralValueType = None,
        ) -> m.Tests.Factory.Entity:
            """Create test entity with identity.

            .. deprecated::
                Use tt.model("entity", name=..., value=...) instead.

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_MODELS_ENTITY,
                DeprecationWarning,
                stacklevel=2,
            )
            # Delegate to DomainHelpers for consistent entity creation
            return u.Tests.DomainHelpers.create_test_entity_instance(
                name=name,
                value=value,
                entity_class=m.Tests.Factory.Entity,
            )

        @staticmethod
        def value_object(
            data: str = c.Tests.Factory.DEFAULT_VALUE_DATA,
            count: int = c.Tests.Factory.DEFAULT_VALUE_COUNT,
        ) -> m.Tests.Factory.ValueObject:
            """Create test value object.

            .. deprecated::
                Use tt.model("value", data=..., count=...) instead.

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_MODELS_VALUE_OBJECT,
                DeprecationWarning,
                stacklevel=2,
            )
            # Delegate to DomainHelpers for consistent value object creation
            return u.Tests.DomainHelpers.create_test_value_object_instance(
                data=data,
                count=count,
                value_class=m.Tests.Factory.ValueObject,
            )

    class Operations:
        """Operation factory methods for callable test functions.

        Creates callable operations for testing handlers, processors, etc.
        """

        @staticmethod
        def simple() -> Callable[[], str]:
            """Create simple operation returning 'success'."""
            warnings.warn(
                c.Tests.Factory.DEPRECATION_OPS_SIMPLE,
                DeprecationWarning,
                stacklevel=2,
            )

            def op() -> str:
                return c.Tests.Factory.SUCCESS_MESSAGE

            return op

        @staticmethod
        def add() -> Callable[
            [t.Tests.TestResultValue, t.Tests.TestResultValue],
            t.Tests.TestResultValue,
        ]:
            """Create add operation for numeric/string concatenation."""
            warnings.warn(
                c.Tests.Factory.DEPRECATION_OPS_ADD,
                DeprecationWarning,
                stacklevel=2,
            )

            def op(
                a: t.Tests.TestResultValue,
                b: t.Tests.TestResultValue,
            ) -> t.Tests.TestResultValue:
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return a + b
                return str(a) + str(b)

            return op

        @staticmethod
        def format() -> Callable[[str, int], str]:
            """Create format operation returning 'name: value'."""
            warnings.warn(
                c.Tests.Factory.DEPRECATION_OPS_FORMAT,
                DeprecationWarning,
                stacklevel=2,
            )

            def op(name: str, value: int = 10) -> str:
                return f"{name}: {value}"

            return op

        @staticmethod
        def error(message: str = c.Tests.Factory.ERROR_DEFAULT) -> Callable[..., Never]:
            """Create operation that raises ValueError."""
            warnings.warn(
                c.Tests.Factory.DEPRECATION_OPS_ERROR,
                DeprecationWarning,
                stacklevel=2,
            )

            def op() -> Never:
                raise ValueError(message)

            return op

        @staticmethod
        def type_error(
            message: str = c.Tests.Factory.ERROR_DEFAULT,
        ) -> Callable[..., Never]:
            """Create operation that raises TypeError."""
            warnings.warn(
                c.Tests.Factory.DEPRECATION_OPS_TYPE_ERROR,
                DeprecationWarning,
                stacklevel=2,
            )

            def op() -> Never:
                raise TypeError(message)

            return op

        @staticmethod
        def result_success[TValue](value: TValue) -> Callable[[], r[TValue]]:
            """Create operation returning success result."""
            warnings.warn(
                c.Tests.Factory.DEPRECATION_OPS_RESULT_OK,
                DeprecationWarning,
                stacklevel=2,
            )

            def op() -> r[TValue]:
                return r[TValue].ok(value)

            return op

        @staticmethod
        def result_failure(
            error: str = c.Tests.Factory.ERROR_OPERATION_FAILED,
        ) -> Callable[[], r[object]]:
            """Create operation returning failure result."""
            warnings.warn(
                c.Tests.Factory.DEPRECATION_OPS_RESULT_FAIL,
                DeprecationWarning,
                stacklevel=2,
            )

            def op() -> r[object]:
                return r[object].fail(error)

            return op

    class Batch:
        """Batch factory methods for creating multiple objects.

        Provides methods for generating lists of test objects.
        """

        @staticmethod
        def users(
            count: int = c.Tests.Factory.DEFAULT_BATCH_COUNT,
        ) -> builtins.list[m.Tests.Factory.User]:
            """Create batch of test users.

            Args:
                count: Number of users to create

            Returns:
                List of user model instances

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_BATCH_USERS,
                DeprecationWarning,
                stacklevel=2,
            )
            return FlextTestsFactories.batch_users(count)

        @staticmethod
        def configs(
            count: int = 3,
            environments: builtins.list[str] | None = None,
        ) -> builtins.list[m.Tests.Factory.Config]:
            """Create batch of test configurations.

            Args:
                count: Number of configs to create
                environments: Optional list of environment names

            Returns:
                List of config model instances

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_BATCH_CONFIGS,
                DeprecationWarning,
                stacklevel=2,
            )
            envs = (
                environments or list(c.Tests.Factory.DEFAULT_BATCH_ENVIRONMENTS)[:count]
            )
            return [
                FlextTestsFactories.create_config(environment=envs[i % len(envs)])
                for i in range(count)
            ]

        @staticmethod
        def services(
            count: int = 3,
            service_types: builtins.list[str] | None = None,
        ) -> builtins.list[m.Tests.Factory.Service]:
            """Create batch of test services.

            Args:
                count: Number of services to create
                service_types: Optional list of service types

            Returns:
                List of service model instances

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_BATCH_SERVICES,
                DeprecationWarning,
                stacklevel=2,
            )
            types = (
                service_types
                or list(c.Tests.Factory.DEFAULT_BATCH_SERVICE_TYPES)[:count]
            )
            return [
                FlextTestsFactories.create_service(service_type=types[i % len(types)])
                for i in range(count)
            ]

        @staticmethod
        def results[TValue](
            values: builtins.list[TValue],
            errors: builtins.list[str] | None = None,
        ) -> builtins.list[r[TValue]]:
            """Create batch of results from values and errors.

            Args:
                values: Values for success results
                errors: Error messages for failure results

            Returns:
                List of FlextResult instances

            """
            warnings.warn(
                c.Tests.Factory.DEPRECATION_BATCH_RESULTS,
                DeprecationWarning,
                stacklevel=2,
            )
            results: builtins.list[r[TValue]] = [r[TValue].ok(v) for v in values]
            if errors:
                results.extend([r[TValue].fail(e) for e in errors])
            return results

    # ==========================================================================
    # STATIC FACTORY METHODS - Primary interface (backward compatible)
    # ==========================================================================

    @staticmethod
    def create_user(
        user_id: str | None = None,
        name: str | None = None,
        email: str | None = None,
        **overrides: t.Tests.TestResultValue,
    ) -> m.Tests.Factory.User:
        """Create a test user.

        Args:
            user_id: Optional user ID
            name: Optional user name
            email: Optional user email
            **overrides: Additional field overrides

        Returns:
            User model instance

        """
        warnings.warn(
            c.Tests.Factory.DEPRECATION_CREATE_USER,
            DeprecationWarning,
            stacklevel=2,
        )
        user_data: t.Types.ConfigurationDict = {
            "id": user_id or u.Tests.Factory.generate_id(),
            "name": name or c.Tests.Factory.DEFAULT_USER_NAME,
            "email": email
            or c.Tests.Factory.user_email(u.Tests.Factory.generate_short_id(8)),
            "active": c.Tests.Factory.DEFAULT_USER_ACTIVE,
        }
        # Convert overrides dict to ConfigurationDict
        overrides_dict: t.Types.ConfigurationDict = dict(overrides)
        merge_result = u.merge(user_data, overrides_dict, strategy="deep")
        if merge_result.is_success:
            user_data = merge_result.value
        return m.Tests.Factory.User.model_validate(user_data)

    @staticmethod
    def create_config(
        service_type: str = c.Tests.Factory.DEFAULT_SERVICE_TYPE,
        environment: str = c.Tests.Factory.DEFAULT_ENVIRONMENT,
        **overrides: t.Tests.TestResultValue,
    ) -> m.Tests.Factory.Config:
        """Create a test configuration.

        Args:
            service_type: Type of service
            environment: Environment name
            **overrides: Additional field overrides

        Returns:
            Config model instance

        """
        warnings.warn(
            c.Tests.Factory.DEPRECATION_CREATE_CONFIG,
            DeprecationWarning,
            stacklevel=2,
        )
        config_data: t.Types.ConfigurationDict = {
            "service_type": service_type,
            "environment": environment,
        }
        # Convert overrides dict to ConfigurationDict
        overrides_dict: t.Types.ConfigurationDict = dict(overrides)
        merge_result = u.merge(config_data, overrides_dict, strategy="deep")
        if merge_result.is_success:
            config_data = merge_result.value
        return m.Tests.Factory.Config.model_validate(config_data)

    @staticmethod
    def create_service(
        service_type: str = c.Tests.Factory.DEFAULT_SERVICE_TYPE,
        service_id: str | None = None,
        **overrides: t.Tests.TestResultValue,
    ) -> m.Tests.Factory.Service:
        """Create a test service.

        Args:
            service_type: Type of service
            service_id: Optional service ID
            **overrides: Additional field overrides

        Returns:
            Service model instance

        """
        warnings.warn(
            c.Tests.Factory.DEPRECATION_CREATE_SERVICE,
            DeprecationWarning,
            stacklevel=2,
        )
        service_data: dict[str, t.Tests.TestResultValue] = {
            "id": service_id or u.generate("uuid"),
            "type": service_type,
            "status": c.Tests.Factory.DEFAULT_SERVICE_STATUS,
        }
        if "name" not in overrides:
            service_data["name"] = c.Tests.Factory.service_name(service_type)
        # Update with compatible types
        service_data.update(dict(overrides))
        return m.Tests.Factory.Service.model_validate(service_data)

    @staticmethod
    def batch_users(
        count: int = c.Tests.Factory.DEFAULT_BATCH_COUNT,
    ) -> builtins.list[m.Tests.Factory.User]:
        """Create a batch of test users.

        Args:
            count: Number of users to create

        Returns:
            List of user model instances

        """
        warnings.warn(
            c.Tests.Factory.DEPRECATION_BATCH_USERS_FUNC,
            DeprecationWarning,
            stacklevel=2,
        )
        return [
            FlextTestsFactories.create_user(
                name=f"User {i}",
                email=f"user{i}@example.com",
            )
            for i in range(count)
        ]

    @classmethod
    def create_test_operation(
        cls,
        operation_type: str = "simple",
        **overrides: t.Tests.TestResultValue,
    ) -> Callable[..., t.Tests.TestResultValue]:
        """Create a test operation callable.

        Args:
            operation_type: Type of operation ('simple', 'add', 'format', 'error')
            **overrides: Additional configuration

        Returns:
            Test operation callable

        """
        warnings.warn(
            c.Tests.Factory.DEPRECATION_CREATE_TEST_OPERATION,
            DeprecationWarning,
            stacklevel=2,
        )
        error_msg = overrides.get("error_message", "Test error")
        # Use isinstance for proper type narrowing (pyrefly compatible)
        error_message: str = error_msg if isinstance(error_msg, str) else "Test error"

        type_error_msg = overrides.get("error_message", "Wrong type")
        # Use isinstance for proper type narrowing (pyrefly compatible)
        type_error_message: str = (
            type_error_msg if isinstance(type_error_msg, str) else "Wrong type"
        )

        ops: dict[str, Callable[..., t.GeneralValueType]] = {
            "simple": u.Tests.Factory.simple_operation,
            "add": u.Tests.Factory.add_operation,
            "format": u.Tests.Factory.format_operation,
            "error": u.Tests.Factory.create_error_operation(error_message),
            "type_error": u.Tests.Factory.create_type_error_operation(
                type_error_message,
            ),
        }
        # Cast to expected return type (GeneralValueType is superset)
        operations = cast("dict[str, Callable[..., t.Tests.TestResultValue]]", ops)

        if operation_type in operations:
            return operations[operation_type]

        def unknown_operation() -> str:
            return f"unknown operation: {operation_type}"

        return unknown_operation

    # ==========================================================================
    # SERVICE FACTORY - For creating dynamic test service classes
    # ==========================================================================

    @staticmethod
    def create_test_service(
        service_type: str = "test",
        **overrides: t.Tests.TestResultValue,
    ) -> type:
        """Create a test service class.

        Args:
            service_type: Type of service to create
            **overrides: Additional attributes for the service

        Returns:
            Test service class

        """
        warnings.warn(
            c.Tests.Factory.DEPRECATION_CREATE_TEST_SERVICE,
            DeprecationWarning,
            stacklevel=2,
        )

        class TestService(s[t_core.GeneralValueType]):
            """Generic test service."""

            name: str | None = None
            amount: int | None = None
            enabled: bool | None = None
            _overrides: dict[str, t.Tests.TestResultValue] = PrivateAttr(
                default_factory=lambda: {},  # noqa: PIE807
            )

            def __init__(
                self,
                **data: (
                    t.ScalarValue
                    | Sequence[t.ScalarValue]
                    | Mapping[str, t.ScalarValue]
                ),
            ) -> None:
                super().__init__(**data)
                # Use object.__setattr__ for frozen Pydantic model with PrivateAttr
                object.__setattr__(self, "_overrides", {**overrides})

            def _validate_name_not_empty(self) -> r[bool]:
                """Validate name is not empty."""
                if self.name is not None and not self.name:
                    return r[bool].fail("Name is required")
                return r[bool].ok(True)

            def _validate_amount_non_negative(self) -> r[bool]:
                """Validate amount is non-negative."""
                if self.amount is not None and self.amount < 0:
                    return r[bool].fail("Amount must be non-negative")
                return r[bool].ok(True)

            def _validate_disabled_without_amount(self) -> r[bool]:
                """Validate disabled service doesn't have amount."""
                has_amount = self.amount is not None and self.amount > 0
                is_disabled = self.enabled is not None and not self.enabled
                if is_disabled and has_amount:
                    return r[bool].fail("Cannot have amount when disabled")
                return r[bool].ok(True)

            def _validate_business_rules_complex(self) -> r[bool]:
                """Validate business rules for complex service."""
                validators = [
                    self._validate_name_not_empty,
                    self._validate_amount_non_negative,
                    self._validate_disabled_without_amount,
                ]
                for validator in validators:
                    result = validator()
                    if result.is_failure:
                        return result
                return r[bool].ok(True)

            def execute(self) -> r[t_core.GeneralValueType]:
                """Execute test operation."""
                if service_type == "user":
                    return u.Tests.Factory.execute_user_service(
                        cast("t.Types.ConfigurationDict", overrides),
                    )
                if service_type == "complex":
                    return u.Tests.Factory.execute_complex_service(
                        self._validate_business_rules_complex(),
                    )
                return u.Tests.Factory.execute_default_service(service_type)

            def validate_business_rules(self) -> r[bool]:
                """Validate business rules for complex service."""
                if service_type == "complex":
                    return self._validate_business_rules_complex()
                return super().validate_business_rules()

            def validate_config(self) -> r[bool]:
                """Validate config for complex service."""
                if service_type != "complex":
                    return r[bool].ok(True)
                if self.name is not None and len(self.name) > 50:
                    return r[bool].fail("Name too long")
                if self.amount is not None and self.amount > 1000:
                    return r[bool].fail("Value too large")
                return r[bool].ok(True)

        return TestService

    # ==========================================================================
    # FLEXTSERVICE INTERFACE IMPLEMENTATION
    # ==========================================================================

    def execute(self) -> r[t.GeneralValueType]:
        """Execute factory service operation.

        Returns default test data when invoked as a service.

        Returns:
            FlextResult with factory name

        """
        return r[t.GeneralValueType].ok("FlextTestsFactories")


tt = FlextTestsFactories  # Preferred short alias
f = FlextTestsFactories  # Alternative alias for compatibility

__all__ = ["FlextTestsFactories", "f", "tt"]
