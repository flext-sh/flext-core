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
from typing import Never, TypeVar

from flext_core import FlextResult, r
from pydantic import BaseModel as _BaseModel

from flext_tests.base import su
from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.typings import t
from flext_tests.utilities import u

TModel = TypeVar("TModel")
TValue = TypeVar("TValue")


class FlextTestsFactories(su[t.GeneralValueType]):
    """Comprehensive test data factories extending FlextService.

    Provides factory methods for creating test objects, services, and domain
    models using the FlextModels foundation. Follows Railway-Oriented Programming
    with FlextResult[T] returns for error-safe test data generation.

    Architecture:
        - Extends FlextService for consistent service patterns
        - Uses FlextModels (m.ValueObject, m.Entity) for domain models
        - Returns FlextResult[T] for operations that can fail
        - Provides both static and instance methods

    Usage:
        # Static factory methods (most common)
        user = FlextTestsFactories.model("user", name="John")
        config = FlextTestsFactories.model("config", debug=True)

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
            # FactoryModel is _BaseModel, so cast is redundant
            return r[t.Tests.Factory.FactoryModel].fail(
                f"Invalid parameters: {params_result.error}",
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
                # Narrow BaseModel to specific types via isinstance
                if isinstance(factory_result, m.Tests.Factory.User):
                    return factory_result
                if isinstance(factory_result, m.Tests.Factory.Config):
                    return factory_result
                if isinstance(factory_result, m.Tests.Factory.Service):
                    return factory_result
                if isinstance(factory_result, m.Tests.Factory.Entity):
                    return factory_result
                if isinstance(factory_result, m.Tests.Factory.ValueObject):
                    return factory_result
                # Fallback for valid BaseModel subclasses
                return m.Tests.Factory.User(
                    id="factory_fallback",
                    name="Factory Result",
                    email="factory@test.com",
                )

            if params.kind == "user":
                user_data: dict[str, t.GeneralValueType] = {
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
                    user_overrides: dict[str, t.GeneralValueType] = dict(
                        params.overrides,
                    )
                    merge_result = u.merge(user_data, user_overrides, strategy="deep")
                    if merge_result.is_success:
                        user_data = merge_result.value
                return m.Tests.Factory.User.model_validate(user_data)

            if params.kind == "config":
                config_data: dict[str, t.GeneralValueType] = {
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
                    config_overrides: dict[str, t.GeneralValueType] = dict(
                        params.overrides,
                    )
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
                    # Type narrowing: params.overrides is Mapping[str, t.GeneralValueType]
                    overrides_mapping = params.overrides
                    overrides_dict: dict[str, t.Tests.TestResultValue] = dict(
                        overrides_mapping,
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
        instance: _BaseModel = _create_single()

        # Apply transform if provided
        if params.transform:
            instance = params.transform(instance)

        # Apply validation if provided
        if params.validate_fn and not params.validate_fn(instance):
            # FactoryModel is BaseModel, so cast is redundant
            return r[t.Tests.Factory.FactoryModel].fail(
                c.Tests.Factory.ERROR_VALIDATION,
            )

        # Handle count > 1
        if params.count > 1:
            # Use list[_BaseModel] since transforms can widen the type
            instances: builtins.list[_BaseModel] = [instance]
            for _ in range(params.count - 1):
                new_instance = _create_single()
                if params.transform:
                    transformed = params.transform(new_instance)
                    # Transform returns _BaseModel, assign directly
                    new_instance_base: _BaseModel = transformed
                    if params.validate_fn and not params.validate_fn(new_instance_base):
                        return r[list[_BaseModel]].fail(
                            c.Tests.Factory.ERROR_VALIDATION,
                        )
                    instances.append(new_instance_base)
                else:
                    if params.validate_fn and not params.validate_fn(new_instance):
                        return r[list[_BaseModel]].fail(
                            c.Tests.Factory.ERROR_VALIDATION,
                        )
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
                        result_dict,
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

            # Return list - instances are already _BaseModel compatible
            typed_instances: list[_BaseModel] = list(instances)
            if params.as_result:
                result: r[list[_BaseModel]] = r.ok(typed_instances)
                return result
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
                str(inst_id): typed_instance,
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
                key: typed_instance,
            }
            if params.as_result:
                single_mapping_result: r[dict[str, _BaseModel]] = r[
                    dict[str, _BaseModel]
                ].ok(single_mapped_dict)
                return single_mapping_result
            return single_mapped_dict

        # Return single instance - already _BaseModel compatible
        if params.as_result:
            return r[_BaseModel].ok(typed_instance)
        return typed_instance

    @classmethod
    def res(
        cls,
        kind: t.Tests.Factory.ResultKind = "ok",
        value: t.GeneralValueType = None,
        # All parameters via kwargs - will be validated by ResultFactoryParams
        **kwargs: t.Tests.TestResultValue,
    ) -> r[t.GeneralValueType] | builtins.list[r[t.GeneralValueType]]:
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
            return r[t.GeneralValueType].fail(
                f"Invalid parameters: {params_result.error}",
            )
        params = params_result.value

        # Handle batch creation with mix_pattern
        if params.mix_pattern is not None and (
            params.values is not None or params.errors is not None
        ):
            result_list: builtins.list[r[t.GeneralValueType]] = []
            val_idx = 0
            err_idx = 0
            for is_success in params.mix_pattern:
                if is_success:
                    if params.values and val_idx < len(params.values):
                        mix_val = params.values[val_idx]
                        if params.transform:
                            mix_val = params.transform(mix_val)
                        result_list.append(r[t.GeneralValueType].ok(mix_val))
                        val_idx += 1
                    elif params.value is not None:
                        mix_val_single: t.GeneralValueType = params.value
                        if params.transform:
                            mix_val_single = params.transform(mix_val_single)
                        result_list.append(r[t.GeneralValueType].ok(mix_val_single))
                elif params.errors and err_idx < len(params.errors):
                    result_list.append(
                        r[t.GeneralValueType].fail(
                            params.errors[err_idx],
                            error_code=params.error_code,
                        ),
                    )
                    err_idx += 1
                else:
                    result_list.append(
                        r[t.GeneralValueType].fail(
                            params.error,
                            error_code=params.error_code,
                        ),
                    )
            return result_list

        # Handle batch creation with explicit values/errors
        if params.values is not None or params.errors is not None or params.count > 1:
            result_list = []
            if params.values:
                for raw_val in params.values:
                    # raw_val is t.GeneralValueType, transform if needed
                    batch_val = (
                        params.transform(raw_val) if params.transform else raw_val
                    )
                    result_list.append(r[t.GeneralValueType].ok(batch_val))
            if params.errors:
                for err in params.errors:
                    result_list.append(
                        r[t.GeneralValueType].fail(err, error_code=params.error_code),
                    )
            if params.count > 1 and not params.values and not params.errors:
                # Create multiple results from single value
                for _ in range(params.count):
                    if params.value is None:
                        # None value always results in failure
                        error_msg = (
                            params.error_on_none or c.Tests.Factory.ERROR_VALUE_NONE
                        )
                        result_list.append(r[t.GeneralValueType].fail(error_msg))
                    else:
                        count_val: t.GeneralValueType = params.value
                        if params.transform:
                            count_val = params.transform(count_val)
                        result_list.append(r[t.GeneralValueType].ok(count_val))
            if result_list:
                return result_list
            # Empty list case
            if params.value is not None:
                empty_case_val = params.value
                return [r[t.GeneralValueType].ok(empty_case_val)]
            return [r[t.GeneralValueType].fail(params.error_on_none or params.error)]

        # Single result creation
        if params.kind == "ok":
            if params.value is None:
                # None value with kind="ok" returns failure with meaningful message
                return r[t.GeneralValueType].fail(
                    params.error_on_none or c.Tests.Factory.ERROR_VALUE_NONE,
                )
            ok_raw = params.value
            ok_transformed = params.transform(ok_raw) if params.transform else ok_raw
            return r[t.GeneralValueType].ok(ok_transformed)

        if params.kind == "fail":
            return r[t.GeneralValueType].fail(
                params.error,
                error_code=params.error_code,
            )

        # params.kind == "from_value"
        if params.value is None:
            return r[t.GeneralValueType].fail(
                params.error_on_none or c.Tests.Factory.ERROR_VALUE_NONE,
            )
        from_val_raw = params.value
        from_val_transformed = (
            params.transform(from_val_raw) if params.transform else from_val_raw
        )
        return r[t.GeneralValueType].ok(from_val_transformed)

    @classmethod
    def op(
        cls,
        kind: t.Tests.Factory.OpKind = "simple",
        *,
        error_message: str = c.Tests.Factory.ERROR_DEFAULT,
        result_value: t.Tests.TestResultValue = c.Tests.Factory.SUCCESS_MESSAGE,
    ) -> object:
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
                user_model = cls.model("user", name=name, email=f"user{i}@example.com")
                # Narrow type via isinstance check
                if isinstance(user_model, m.Tests.Factory.User):
                    result_users.append(user_model)
            return result_users

        if kind == "config":
            envs = (
                list(environments)
                if environments
                else list(c.Tests.Factory.DEFAULT_BATCH_ENVIRONMENTS)
            )
            configs: builtins.list[m.Tests.Factory.Config] = []
            for i in range(count):
                config_model = cls.model("config", environment=envs[i % len(envs)])
                # Narrow type via isinstance check
                if isinstance(config_model, m.Tests.Factory.Config):
                    configs.append(config_model)
            return configs

        # kind == "service"
        types = (
            list(service_types)
            if service_types
            else list(c.Tests.Factory.DEFAULT_BATCH_SERVICE_TYPES)
        )
        services: builtins.list[m.Tests.Factory.Service] = []
        for i in range(count):
            service_model = cls.model("service", service_type=types[i % len(types)])
            # Narrow type via isinstance check
            if isinstance(service_model, m.Tests.Factory.Service):
                services.append(service_model)
        return services

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
    def list(
        cls,
        source: (
            Sequence[t.GeneralValueType]
            | Callable[[], t.GeneralValueType]
            | t.Tests.Factory.ModelKind
        ) = "user",
        # All parameters via kwargs - will be validated by ListFactoryParams
        **kwargs: t.Tests.TestResultValue,
    ) -> builtins.list[t.GeneralValueType] | r[builtins.list[t.GeneralValueType]]:
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
            return r[builtins.list[t.GeneralValueType]].fail(
                f"Invalid parameters: {params_result.error}",
            )
        params = params_result.value

        items: builtins.list[t.GeneralValueType] = []

        # Handle different source types - use list[t.GeneralValueType] for intermediate storage
        # then convert to list[T] at the end for proper type inference
        raw_items: builtins.list[t.GeneralValueType] = []

        if isinstance(params.source, str):
            # ModelKind string - delegate to tt.model()
            model_kind_str = params.source
            # Use Literal mapping for type-safe conversion
            kind_map: dict[str, t.Tests.Factory.ModelKind] = {
                "user": "user",
                "config": "config",
                "service": "service",
                "entity": "entity",
                "value": "value",
                "command": "command",
                "query": "query",
                "event": "event",
            }
            if model_kind_str not in kind_map:
                return r[builtins.list[t.GeneralValueType]].fail(
                    f"Invalid model kind: {model_kind_str}",
                )
            model_kind = kind_map[model_kind_str]
            for _ in range(params.count):
                model_result = cls.model(model_kind)
                # Extract item from model result
                raw_item: t.GeneralValueType
                if isinstance(model_result, list) and model_result:
                    raw_item = model_result[0]
                elif isinstance(model_result, FlextResult):
                    if model_result.is_success:
                        raw_item = model_result.value
                    else:
                        continue
                else:
                    raw_item = u.Conversion.to_general_value_type(model_result)
                # Apply transform and filter
                if params.transform:
                    raw_item = params.transform(raw_item)
                if params.filter_ is None or params.filter_(raw_item):
                    raw_items.append(raw_item)

        elif callable(params.source):
            source_callable: Callable[[], t.GeneralValueType] = params.source
            for _ in range(params.count):
                raw_item = source_callable()
                if params.transform:
                    raw_item = params.transform(raw_item)
                if params.filter_ is None or params.filter_(raw_item):
                    raw_items.append(raw_item)

        elif u.is_type(params.source, "sequence_not_str"):
            # Sequence - use items directly (exclude strings)
            source_seq: Sequence[t.GeneralValueType] = params.source
            for source_item in source_seq:
                final_item: t.GeneralValueType
                if params.transform:
                    final_item = params.transform(source_item)
                else:
                    final_item = source_item
                if params.filter_ is None or params.filter_(final_item):
                    raw_items.append(final_item)

        # Transfer raw_items to items - type is enforced by caller's usage
        items.extend(raw_items)  # PERF402: use extend instead of loop

        # Ensure uniqueness if requested
        if params.unique and items:
            seen: set[object] = set()
            unique_items: builtins.list[t.GeneralValueType] = []
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
            return r[builtins.list[t.GeneralValueType]].ok(items)
        return items

    @classmethod
    def dict_factory(
        cls,
        source: (
            Mapping[str, t.GeneralValueType]
            | Callable[[], tuple[str, t.GeneralValueType]]
            | t.Tests.Factory.ModelKind
        ) = "user",
        # All parameters via kwargs - will be validated by DictFactoryParams
        **kwargs: t.Tests.TestResultValue,
    ) -> dict[str, t.GeneralValueType] | r[dict[str, t.GeneralValueType]]:
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
            return r[dict[str, t.GeneralValueType]].fail(
                f"Invalid parameters: {params_result.error}",
            )
        params = params_result.value

        result_dict: dict[str, t.GeneralValueType] = {}

        # Handle different source types
        if isinstance(params.source, str):
            # ModelKind - create models with auto keys
            # Use default "user" for type safety, actual source is validated at model()
            model_kind: t.Tests.Factory.ModelKind = "user"
            # Override if source matches known kinds
            match params.source:
                case "user":
                    model_kind = "user"
                case "config":
                    model_kind = "config"
                case "service":
                    model_kind = "service"
                case "entity":
                    model_kind = "entity"
                case "value":
                    model_kind = "value"
                case "command":
                    model_kind = "command"
                case "query":
                    model_kind = "query"
                case "event":
                    model_kind = "event"
                case _:
                    model_kind = "user"
            for i in range(params.count):
                key: str = params.key_factory(i) if params.key_factory else f"item_{i}"

                # Create model instance
                model_result = cls.model(model_kind, count=1)
                # Extract value from model result
                value: t.GeneralValueType
                if isinstance(model_result, list) and model_result:
                    value = model_result[0]
                elif isinstance(model_result, FlextResult):
                    # isinstance() narrows to FlextResult[...], has is_success and value
                    if model_result.is_success:
                        value = model_result.value
                    else:
                        continue  # Skip failed results
                else:
                    value = u.Conversion.to_general_value_type(model_result)

                if params.value_factory:
                    value = params.value_factory(key)

                result_dict[key] = value

        elif callable(params.source):
            # Callable factory returning (key, value) tuples
            source_callable = params.source
            for i in range(params.count):
                key_val, value_val = source_callable()
                call_key = key_val
                call_value = value_val
                if params.key_factory:
                    call_key = params.key_factory(i)
                if params.value_factory:
                    call_value = params.value_factory(call_key)
                result_dict[call_key] = call_value

        elif u.is_type(params.source, "mapping") and not u.is_type(
            params.source,
            "str",
        ):
            # Mapping - use directly (exclude strings)
            source_mapping = params.source
            for k, v in source_mapping.items():
                result_dict[str(k)] = v

        # Merge with additional mapping
        if params.merge_with:
            merge_mapping = params.merge_with
            for k, v in merge_mapping.items():
                result_dict[str(k)] = v

        if params.as_result:
            return r[dict[str, t.GeneralValueType]].ok(result_dict)
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
            # Type annotation - params.type_ is validated to be type[T]
            type_cls: type[T] = params.type_
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
                            f"Failed to create instance: {e}",
                        )
                    raise

            if params.as_result:
                return r[builtins.list[T]].ok(instances)
            return instances

        # Single instance
        try:
            instance = _create_instance()
            if params.as_result:
                result_instance: r[T] = r.ok(instance)
                return result_instance
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
        return cls._create_test_service_impl(kind, **overrides)

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
        user_data: dict[str, t.GeneralValueType] = {
            "id": user_id or u.Tests.Factory.generate_id(),
            "name": name or c.Tests.Factory.DEFAULT_USER_NAME,
            "email": email
            or c.Tests.Factory.user_email(u.Tests.Factory.generate_short_id(8)),
            "active": c.Tests.Factory.DEFAULT_USER_ACTIVE,
        }
        # Convert overrides dict to ConfigurationDict
        overrides_dict: dict[str, t.GeneralValueType] = dict(overrides)
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
        config_data: dict[str, t.GeneralValueType] = {
            "service_type": service_type,
            "environment": environment,
        }
        # Convert overrides dict to ConfigurationDict
        overrides_dict: dict[str, t.GeneralValueType] = dict(overrides)
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
        users: list[m.Tests.Factory.User] = []
        for i in range(count):
            result = FlextTestsFactories.model(
                "user",
                name=f"User {i}",
                email=f"user{i}@example.com",
            )
            # Extract user from the result
            if isinstance(result, m.Tests.Factory.User):
                users.append(result)
            elif isinstance(result, list) and result:
                first_item = result[0]
                if isinstance(first_item, m.Tests.Factory.User):
                    users.append(first_item)
        return users

    @classmethod
    def create_test_operation(
        cls,
        operation_type: str = "simple",
        **overrides: t.Tests.TestResultValue,
    ) -> object:
        """Create a test operation callable.

        Args:
            operation_type: Type of operation ('simple', 'add', 'format', 'error')
            **overrides: Additional configuration

        Returns:
            Test operation callable (object - varying signatures)

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

        ops: dict[str, object] = {
            "simple": u.Tests.Factory.simple_operation,
            "add": u.Tests.Factory.add_operation,
            "format": u.Tests.Factory.format_operation,
            "error": u.Tests.Factory.create_error_operation(error_message),
            "type_error": u.Tests.Factory.create_type_error_operation(
                type_error_message,
            ),
        }
        operations = ops

        if operation_type in operations:
            return operations[operation_type]

        def unknown_operation() -> str:
            return f"unknown operation: {operation_type}"

        return unknown_operation

    # ==========================================================================
    # SERVICE FACTORY - For creating dynamic test service classes
    # ==========================================================================

    @staticmethod
    def _create_test_service_impl(
        service_type: str = "test",
        **overrides: t.Tests.TestResultValue,
    ) -> type:
        """Internal implementation for creating test service classes.

        This is the actual implementation used by both svc() and
        create_test_service(). The public create_test_service() method
        emits a deprecation warning before calling this.

        Args:
            service_type: Type of service to create
            **overrides: Additional attributes for the service

        Returns:
            Test service class

        """
        # Capture overrides in local variable for use in nested class
        captured_overrides: dict[str, t.Tests.TestResultValue] = dict(overrides)

        class TestService(su[t.GeneralValueType]):
            """Generic test service."""

            name: str | None = None
            amount: int | None = None
            enabled: bool | None = None
            # Use class attribute (not PrivateAttr) to match FlextService pattern
            # Initialize as None to avoid ClassVar requirement (mutable default)
            _overrides: dict[str, t.Tests.TestResultValue] | None = None

            def __init__(
                self,
                **data: (
                    t.ScalarValue
                    | Sequence[t.ScalarValue]
                    | Mapping[str, t.ScalarValue]
                ),
            ) -> None:
                # Separate service fields from overrides
                # Service fields: name, amount, enabled (for complex service)
                # Overrides: default, etc. (for execute methods)
                override_fields: dict[str, t.GeneralValueType] = {}
                # Extract service fields directly to avoid mypy dict unpacking issues
                # Build kwargs inline to match **kwargs signature
                name_value: t.GeneralValueType | None = None
                amount_value: t.GeneralValueType | None = None
                enabled_value: t.GeneralValueType | None = None

                # Use captured_overrides from outer scope (closure)
                for key, value in {**captured_overrides, **data}.items():
                    # value is already t.GeneralValueType from dict iteration
                    gv: t.GeneralValueType = value
                    if key == "name":
                        name_value = gv
                    elif key == "amount":
                        amount_value = gv
                    elif key == "enabled":
                        enabled_value = gv
                    else:
                        override_fields[key] = gv

                # Call parent with **data dict (FlextService.__init__ accepts **data: t.GeneralValueType)
                # Build service data dict with only non-None values
                # Type compatibility: t.GeneralValueType is compatible with t.GeneralValueType
                # (both are from flext_core.typings, t is just an alias)
                service_data: dict[str, t.GeneralValueType] = {}
                if name_value is not None:
                    service_data["name"] = name_value
                if amount_value is not None:
                    service_data["amount"] = amount_value
                if enabled_value is not None:
                    service_data["enabled"] = enabled_value
                # Call parent with **service_data unpacking
                # MyPy limitation: dict unpacking to **kwargs not fully supported for BaseModel.__init__
                # The dict is compatible at runtime, but MyPy can't infer the type compatibility
                # Solution: Use cast to help MyPy understand the type compatibility
                # BaseModel.__init__ accepts **data: t.GeneralValueType, so this is safe
                super().__init__(**service_data)
                # Set attribute directly (no PrivateAttr needed, compatible with FlextService)
                # Initialize mutable attribute in __init__ to avoid ClassVar requirement
                # Type narrowing: override_fields is dict[str, t.GeneralValueType]
                # but _overrides expects dict[str, t.Tests.TestResultValue]
                # Convert to TestResultValue type
                self._overrides = override_fields

            def _validate_name_not_empty(self) -> r[bool]:
                """Validate name is not empty (only if name is provided)."""
                # Only validate if name was explicitly provided (not None)
                # If name is None, it means it wasn't provided, so skip validation
                # If name is provided but empty, fail validation
                if self.name is not None and not self.name:
                    return r[bool].fail("Name is required")
                return r[bool].ok(value=True)

            def _validate_amount_non_negative(self) -> r[bool]:
                """Validate amount is non-negative."""
                if self.amount is not None and self.amount < 0:
                    return r[bool].fail("Amount must be non-negative")
                return r[bool].ok(value=True)

            def _validate_disabled_without_amount(self) -> r[bool]:
                """Validate disabled service doesn't have amount."""
                has_amount = self.amount is not None and self.amount > 0
                is_disabled = self.enabled is not None and not self.enabled
                if is_disabled and has_amount:
                    return r[bool].fail("Cannot have amount when disabled")
                return r[bool].ok(value=True)

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
                return r[bool].ok(value=True)

            def execute(self) -> r[t.GeneralValueType]:
                """Execute test operation."""
                if service_type == "user":
                    # Merge instance overrides with class overrides
                    merged_overrides = {**overrides}
                    if self._overrides is not None:
                        merged_overrides.update(self._overrides)
                    return u.Tests.Factory.execute_user_service(
                        merged_overrides,
                    )
                if service_type == "complex":
                    validation_result = self._validate_business_rules_complex()
                    # Use concrete FlextResult type instead of protocol
                    return u.Tests.Factory.execute_complex_service(validation_result)
                return u.Tests.Factory.execute_default_service(service_type)

            def validate_business_rules(self) -> r[bool]:
                """Validate business rules for complex service."""
                if service_type == "complex":
                    return self._validate_business_rules_complex()
                return super().validate_business_rules()

            def validate_config(self) -> r[bool]:
                """Validate config for complex service."""
                if service_type != "complex":
                    return r[bool].ok(value=True)
                if self.name is not None and len(self.name) > 50:
                    return r[bool].fail("Name too long")
                if self.amount is not None and self.amount > 1000:
                    return r[bool].fail("Value too large")
                return r[bool].ok(value=True)

        return TestService

    @staticmethod
    def create_test_service(
        service_type: str = "test",
        **overrides: t.Tests.TestResultValue,
    ) -> type:
        """Create a test service class.

        .. deprecated::
            Use tt.svc() instead. This method will be removed in a future version.

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
        return FlextTestsFactories._create_test_service_impl(service_type, **overrides)

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
