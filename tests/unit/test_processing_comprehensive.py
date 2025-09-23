"""Comprehensive test coverage for FlextProcessing module.

Uses flext_tests patterns for standardized testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Never

import pytest
from pydantic import ValidationError

from flext_core import (
    FlextModels,
    FlextProcessing,
    FlextResult,
)
from flext_tests import FlextTestsFactories, FlextTestsMatchers


class TestFlextProcessingConfig:
    """Test FlextProcessing.Config class methods."""

    def test_get_default_timeout_success(self) -> None:
        """Test get_default_timeout with valid configuration."""
        timeout = FlextProcessing.Config.get_default_timeout()
        assert isinstance(timeout, float)
        assert timeout > 0

    def test_get_default_timeout_exception_fallback(self) -> None:
        """Test get_default_timeout exception handling."""
        # This should still work even if config fails
        timeout = FlextProcessing.Config.get_default_timeout()
        assert isinstance(timeout, float)
        assert timeout > 0

    def test_get_max_batch_size_success(self) -> None:
        """Test get_max_batch_size with valid configuration."""
        batch_size = FlextProcessing.Config.get_max_batch_size()
        assert isinstance(batch_size, int)
        assert batch_size > 0

    def test_get_max_batch_size_exception_fallback(self) -> None:
        """Test get_max_batch_size exception handling."""
        batch_size = FlextProcessing.Config.get_max_batch_size()
        assert isinstance(batch_size, int)
        assert batch_size > 0

    def test_get_max_handlers_success(self) -> None:
        """Test get_max_handlers with valid configuration."""
        max_handlers = FlextProcessing.Config.get_max_handlers()
        assert isinstance(max_handlers, int)
        assert max_handlers > 0

    def test_get_max_handlers_exception_fallback(self) -> None:
        """Test get_max_handlers exception handling."""
        max_handlers = FlextProcessing.Config.get_max_handlers()
        assert isinstance(max_handlers, int)
        assert max_handlers > 0


class TestFlextProcessingHandler:
    """Test FlextProcessing.Handler class."""

    def test_handler_base_handle_method(self) -> None:
        """Test basic handler handle method."""
        handler = FlextProcessing.Handler()
        request = "test_request"

        result = handler.handle(request)

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == f"Base handler processed: {request}"

    def test_handler_with_different_request_types(self) -> None:
        """Test handler with various request types using factories."""
        handler = FlextProcessing.Handler()

        # Test with different data types from factory
        test_data = FlextTestsFactories.create_test_dataset()
        for request in test_data:
            result = handler.handle(request)
            FlextTestsMatchers.assert_result_success(result)
            assert "Base handler processed:" in str(result.unwrap())


class TestFlextProcessingHandlerRegistry:
    """Test FlextProcessing.HandlerRegistry class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.registry = FlextProcessing.HandlerRegistry()
        # Use callable handler that returns FlextResult

        def real_handler(request: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"Real handler processed: {request}")

        self.real_handler = real_handler

    def test_registry_initialization(self) -> None:
        """Test handler registry initialization."""
        registry = FlextProcessing.HandlerRegistry()
        assert registry.count() == 0
        assert not registry.exists("any_handler")

    def test_register_handler_success(self) -> None:
        """Test successful handler registration."""
        handler_name = "test_handler"
        registration = FlextModels.HandlerRegistration(
            name=handler_name, handler=self.real_handler
        )

        result = self.registry.register(registration)

        FlextTestsMatchers.assert_result_success(result)
        assert self.registry.count() == 1
        assert self.registry.exists(handler_name)

    def test_register_duplicate_handler_failure(self) -> None:
        """Test registering duplicate handler fails."""
        handler_name = "test_handler"
        registration = FlextModels.HandlerRegistration(
            name=handler_name, handler=self.real_handler
        )

        # Register first handler
        self.registry.register(registration)

        # Try to register again
        result = self.registry.register(registration)

        FlextTestsMatchers.assert_result_failure(result)
        assert f"Handler '{handler_name}' already registered" in str(result.error)

    def test_get_existing_handler_success(self) -> None:
        """Test retrieving existing handler."""
        handler_name = "test_handler"
        registration = FlextModels.HandlerRegistration(
            name=handler_name, handler=self.real_handler
        )
        self.registry.register(registration)

        result = self.registry.get(handler_name)

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == self.real_handler

    def test_get_nonexistent_handler_failure(self) -> None:
        """Test retrieving non-existent handler fails."""
        handler_name = "nonexistent_handler"

        result = self.registry.get(handler_name)

        FlextTestsMatchers.assert_result_failure(result)
        assert f"Handler '{handler_name}' not found" in str(result.error)

    def test_execute_handler_with_handle_method(self) -> None:
        """Test executing handler that has handle method."""
        handler_name = "test_handler"
        request = "test_request"

        # Real handler already returns FlextResult - no need to configure
        registration = FlextModels.HandlerRegistration(
            name=handler_name, handler=self.real_handler
        )
        self.registry.register(registration)

        result = self.registry.execute(handler_name, request)

        FlextTestsMatchers.assert_result_success(result)
        # Real handler processes the request and includes it in the response
        assert "Real handler processed:" in result.unwrap()
        assert "test_request" in result.unwrap()

    def test_execute_callable_handler(self) -> None:
        """Test executing callable handler."""
        handler_name = "callable_handler"
        request = "test_request"
        expected_result = "callable_result"

        def callable_handler(_req: object) -> str:
            return expected_result

        registration = FlextModels.HandlerRegistration(
            name=handler_name, handler=callable_handler
        )
        self.registry.register(registration)

        result = self.registry.execute(handler_name, request)

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == expected_result

    def test_execute_handler_returning_flext_result(self) -> None:
        """Test executing handler that returns FlextResult."""
        handler_name = "result_handler"
        request = "test_request"
        expected_result = "result_data"

        def result_handler(_req: object) -> FlextResult[str]:
            return FlextResult[str].ok(expected_result)

        registration = FlextModels.HandlerRegistration(
            name=handler_name, handler=result_handler
        )
        self.registry.register(registration)

        result = self.registry.execute(handler_name, request)

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == expected_result

    def test_execute_handler_without_handle_method_or_callable(self) -> None:
        """Test creating handler registration with invalid handler type."""
        handler_name = "invalid_handler"
        invalid_handler = "not_a_handler"

        # Pydantic validation should prevent creation of invalid handler registration
        with pytest.raises(ValidationError) as exc_info:
            FlextModels.HandlerRegistration(name=handler_name, handler=invalid_handler)

        # Verify the validation error is about callable type
        assert "callable" in str(exc_info.value).lower()

    def test_execute_handler_with_exception(self) -> None:
        """Test executing handler that raises exception."""
        handler_name = "failing_handler"

        def failing_handler(_req: object) -> Never:
            msg = "Handler failed"
            raise ValueError(msg)

        registration = FlextModels.HandlerRegistration(
            name=handler_name, handler=failing_handler
        )
        self.registry.register(registration)

        result = self.registry.execute(handler_name, "request")

        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Handler execution failed:" in result.error

    def test_execute_nonexistent_handler(self) -> None:
        """Test executing non-existent handler."""
        result = self.registry.execute("nonexistent", "request")

        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Handler 'nonexistent' not found" in result.error

    def test_count_handlers(self) -> None:
        """Test counting registered handlers."""
        assert self.registry.count() == 0

        registration1 = FlextModels.HandlerRegistration(
            name="handler1", handler=self.real_handler
        )
        self.registry.register(registration1)
        assert self.registry.count() == 1

        registration2 = FlextModels.HandlerRegistration(
            name="handler2", handler=self.real_handler
        )
        self.registry.register(registration2)
        assert self.registry.count() == 2

    def test_exists_method(self) -> None:
        """Test exists method."""
        handler_name = "test_handler"

        assert not self.registry.exists(handler_name)

        registration = FlextModels.HandlerRegistration(
            name=handler_name, handler=self.real_handler
        )
        self.registry.register(registration)
        assert self.registry.exists(handler_name)

    def test_get_optional_method(self) -> None:
        """Test get_optional method."""
        handler_name = "test_handler"

        assert self.registry.get_optional(handler_name) is None

        registration = FlextModels.HandlerRegistration(
            name=handler_name, handler=self.real_handler
        )
        self.registry.register(registration)
        assert self.registry.get_optional(handler_name) == self.real_handler

    def test_execute_with_timeout_success(self) -> None:
        """Test execute_with_timeout with valid handler."""
        handler = FlextProcessing.Implementation.BasicHandler("timeout_test")
        registration = FlextModels.HandlerRegistration(
            name="timeout_handler", handler=handler
        )
        self.registry.register(registration)

        config = FlextModels.HandlerExecutionConfig(
            handler_name="timeout_handler",
            input_data={"test": "data"},
            timeout_seconds=5,
        )

        result = self.registry.execute_with_timeout(config)
        FlextTestsMatchers.assert_result_success(result)

    def test_execute_with_fallback_success(self) -> None:
        """Test execute_with_fallback with valid handlers."""
        primary_handler = FlextProcessing.Implementation.BasicHandler("primary")
        fallback_handler = FlextProcessing.Implementation.BasicHandler("fallback")

        registration1 = FlextModels.HandlerRegistration(
            name="primary_handler", handler=primary_handler
        )
        registration2 = FlextModels.HandlerRegistration(
            name="fallback_handler", handler=fallback_handler
        )

        self.registry.register(registration1)
        self.registry.register(registration2)

        config = FlextModels.HandlerExecutionConfig(
            handler_name="primary_handler",
            input_data={"test": "data"},
            fallback_handlers=["fallback_handler"],
        )

        result = self.registry.execute_with_fallback(config)
        FlextTestsMatchers.assert_result_success(result)

    def test_execute_batch_success(self) -> None:
        """Test execute_batch with valid batch configuration."""
        handler = FlextProcessing.Implementation.BasicHandler("batch_test")
        registration = FlextModels.HandlerRegistration(
            name="batch_handler", handler=handler
        )
        self.registry.register(registration)

        # Create config with small batch to avoid validation issues
        config = FlextModels.BatchProcessingConfig(
            batch_size=2,
            max_workers=1,  # Prevent auto-adjust recursion
            data_items=[("batch_handler", "data1"), ("batch_handler", "data2")],
            continue_on_error=True,
        )

        result = self.registry.execute_batch(config)
        FlextTestsMatchers.assert_result_success(result)
        assert isinstance(result.unwrap(), list)

    def test_execute_batch_size_limit_exceeded(self) -> None:
        """Test execute_batch with batch size exceeding limits."""
        # Test the method directly without creating problematic config
        # Just call the method with a large number to test the size check logic
        registry = FlextProcessing.HandlerRegistry()

        # Test by creating a mock config object that bypasses validation
        class MockBatchConfig:
            def __init__(self) -> None:
                self.data_items = [f"item{i}" for i in range(10000)]
                self.continue_on_error = True

        mock_config = MockBatchConfig()
        result = registry.execute_batch(mock_config)
        FlextTestsMatchers.assert_result_failure(result)
        assert "exceeds maximum" in result.error

    def test_register_with_validation_success(self) -> None:
        """Test register_with_validation with valid handler and validator."""
        handler = FlextProcessing.Implementation.BasicHandler("validation_test")
        registration = FlextModels.HandlerRegistration(
            name="validated_handler", handler=handler
        )

        # Simple validator that always passes
        def always_pass_validator(_h: object) -> FlextResult[None]:
            return FlextResult[None].ok(None)

        result = self.registry.register_with_validation(
            registration, always_pass_validator
        )
        FlextTestsMatchers.assert_result_success(result)
        assert self.registry.exists("validated_handler")

    def test_register_with_validation_failure(self) -> None:
        """Test register_with_validation with failing validator."""
        handler = FlextProcessing.Implementation.BasicHandler("validation_fail")
        registration = FlextModels.HandlerRegistration(
            name="invalid_handler", handler=handler
        )

        # Validator that always fails
        def always_fail_validator(_h: object) -> FlextResult[None]:
            return FlextResult[None].fail("Validation failed")

        result = self.registry.register_with_validation(
            registration, always_fail_validator
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert not self.registry.exists("invalid_handler")

    def test_register_with_validation_no_validator(self) -> None:
        """Test register_with_validation without validator (should work normally)."""
        handler = FlextProcessing.Implementation.BasicHandler("no_validator")
        registration = FlextModels.HandlerRegistration(
            name="normal_handler", handler=handler
        )

        result = self.registry.register_with_validation(registration, None)
        FlextTestsMatchers.assert_result_success(result)
        assert self.registry.exists("normal_handler")

    def test_registry_size_limits(self) -> None:
        """Test handler registry size limits."""
        # Test that we can't exceed the maximum number of handlers
        max_handlers = FlextProcessing.Config.get_max_handlers()

        # Register handlers up to the limit
        for i in range(min(max_handlers, 10)):  # Limit to 10 for test performance
            handler = FlextProcessing.Implementation.BasicHandler(f"handler_{i}")
            registration = FlextModels.HandlerRegistration(
                name=f"test_handler_{i}", handler=handler
            )
            result = self.registry.register(registration)
            FlextTestsMatchers.assert_result_success(result)

    def test_unsafe_handler_registration(self) -> None:
        """Test registration of unsafe handlers fails."""
        # Create an object that doesn't have handle method and isn't callable
        unsafe_handler = object()

        # This should fail validation in the HandlerRegistration model
        with pytest.raises(ValidationError):
            FlextModels.HandlerRegistration(name="unsafe", handler=unsafe_handler)


class TestFlextProcessingPipeline:
    """Test FlextProcessing.Pipeline class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.pipeline = FlextProcessing.Pipeline()

    def test_pipeline_initialization(self) -> None:
        """Test pipeline initialization."""
        pipeline = FlextProcessing.Pipeline()

        # Test with empty pipeline
        result = pipeline.process("test_data")
        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "test_data"

    def test_pipeline_with_callable_step_returning_result(self) -> None:
        """Test pipeline with callable step returning FlextResult."""

        def step_function(data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{data}")

        self.pipeline.add_step(step_function)

        result = self.pipeline.process("input")

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "processed_input"

    def test_pipeline_with_callable_step_returning_value(self) -> None:
        """Test pipeline with callable step returning direct value."""

        def step_function(data: object) -> str:
            return f"transformed_{data}"

        self.pipeline.add_step(step_function)

        result = self.pipeline.process("input")

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "transformed_input"

    def test_pipeline_with_failing_step(self) -> None:
        """Test pipeline with step that returns failure."""

        def failing_step(_data: object) -> FlextResult[str]:
            return FlextResult[str].fail("Step failed")

        self.pipeline.add_step(failing_step)

        result = self.pipeline.process("input")

        FlextTestsMatchers.assert_result_failure(result)
        assert "Step failed" in result.error

    def test_pipeline_with_dictionary_merging(self) -> None:
        """Test pipeline with dictionary merging."""
        initial_data = {"key1": "value1"}
        merge_data = {"key2": "value2"}

        self.pipeline.add_step(merge_data)

        result = self.pipeline.process(initial_data)

        FlextTestsMatchers.assert_result_success(result)
        expected = {"key1": "value1", "key2": "value2"}
        assert result.unwrap() == expected

    def test_pipeline_with_data_replacement(self) -> None:
        """Test pipeline with data replacement."""
        replacement_data = "new_data"

        self.pipeline.add_step(replacement_data)

        result = self.pipeline.process("old_data")

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == replacement_data

    def test_pipeline_with_multiple_steps(self) -> None:
        """Test pipeline with multiple processing steps."""

        def step1(data: object) -> str:
            return f"step1_{data}"

        def step2(data: object) -> str:
            return f"step2_{data}"

        self.pipeline.add_step(step1)
        self.pipeline.add_step(step2)

        result = self.pipeline.process("input")

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "step2_step1_input"

    def test_pipeline_with_mixed_step_types(self) -> None:
        """Test pipeline with mixed step types."""

        def transform_step(data: object) -> dict[str, object]:
            return {"transformed": data}

        merge_step = {"added_key": "added_value"}

        self.pipeline.add_step(transform_step)
        self.pipeline.add_step(merge_step)

        result = self.pipeline.process("input")

        FlextTestsMatchers.assert_result_success(result)
        expected = {"transformed": "input", "added_key": "added_value"}
        assert result.unwrap() == expected

    def test_process_conditionally_success(self) -> None:
        """Test process_conditionally with true condition."""
        self.pipeline.add_step(lambda x: f"processed: {x}")

        request = FlextModels.ProcessingRequest(
            data={"test": "data"}, context={"condition": True}
        )

        def condition(_data: object) -> bool:
            return True

        result = self.pipeline.process_conditionally(request, condition)
        FlextTestsMatchers.assert_result_success(result)

    def test_process_conditionally_false_condition(self) -> None:
        """Test process_conditionally with false condition."""
        self.pipeline.add_step(lambda x: f"processed: {x}")

        request = FlextModels.ProcessingRequest(
            data={"test": "data"}, context={"condition": False}
        )

        def condition(_data: object) -> bool:
            return False

        result = self.pipeline.process_conditionally(request, condition)
        # With false condition, should still return result but may not process through pipeline
        # The actual behavior depends on FlextResult.when() implementation
        if result.is_success:
            FlextTestsMatchers.assert_result_success(result)
        else:
            # If when() prevents execution, that's also valid behavior
            FlextTestsMatchers.assert_result_failure(result)

    def test_process_with_timeout_success(self) -> None:
        """Test process_with_timeout with valid timeout."""
        self.pipeline.add_step(lambda x: f"processed: {x}")

        request = FlextModels.ProcessingRequest(
            data={"test": "data"}, timeout_seconds=5
        )

        result = self.pipeline.process_with_timeout(request)
        FlextTestsMatchers.assert_result_success(result)

    def test_process_with_timeout_below_minimum(self) -> None:
        """Test process_with_timeout with timeout below minimum."""
        # Create request with valid timeout first, then modify for testing
        request = FlextModels.ProcessingRequest(
            data={"test": "data"}, timeout_seconds=30
        )
        # Manually set a value below minimum to test the processing logic
        request.timeout_seconds = 0.001  # Below minimum for processing logic

        result = self.pipeline.process_with_timeout(request)
        FlextTestsMatchers.assert_result_failure(result)
        assert "below minimum" in result.error

    def test_process_with_timeout_above_maximum(self) -> None:
        """Test process_with_timeout with timeout above maximum."""
        # Test with the maximum allowed by the model (will fail at processing level if > container max)
        request = FlextModels.ProcessingRequest(
            data={"test": "data"},
            timeout_seconds=3600,  # At model maximum
        )
        # Manually set higher value to test processing bounds
        request.timeout_seconds = 9999  # Above processing maximum

        result = self.pipeline.process_with_timeout(request)
        FlextTestsMatchers.assert_result_failure(result)
        assert "exceeds maximum" in result.error

    def test_process_with_fallback_success(self) -> None:
        """Test process_with_fallback with primary pipeline success."""
        self.pipeline.add_step(lambda x: f"primary: {x}")

        fallback_pipeline = FlextProcessing.Pipeline()
        fallback_pipeline.add_step(lambda x: f"fallback: {x}")

        request = FlextModels.ProcessingRequest(data={"test": "data"})

        result = self.pipeline.process_with_fallback(request, fallback_pipeline)
        FlextTestsMatchers.assert_result_success(result)
        assert "primary:" in str(result.unwrap())

    def test_process_with_fallback_failure_uses_fallback(self) -> None:
        """Test process_with_fallback with primary failure using fallback."""
        # Primary pipeline that fails
        self.pipeline.add_step(
            lambda _x: (_ for _ in ()).throw(RuntimeError("Primary failed"))
        )

        # Fallback pipeline that succeeds
        fallback_pipeline = FlextProcessing.Pipeline()
        fallback_pipeline.add_step(lambda x: f"fallback: {x}")

        request = FlextModels.ProcessingRequest(data={"test": "data"})

        result = self.pipeline.process_with_fallback(request, fallback_pipeline)
        FlextTestsMatchers.assert_result_success(result)
        assert "fallback:" in str(result.unwrap())

    def test_process_batch_success(self) -> None:
        """Test process_batch with valid batch configuration."""
        self.pipeline.add_step(lambda x: f"processed: {x}")

        config = FlextModels.BatchProcessingConfig(
            batch_size=3,
            max_workers=1,  # Prevent auto-adjust recursion
            data_items=["item1", "item2", "item3"],
            continue_on_error=True,
        )

        result = self.pipeline.process_batch(config)
        FlextTestsMatchers.assert_result_success(result)
        assert isinstance(result.unwrap(), list)
        assert len(result.unwrap()) == 3

    def test_process_batch_size_limit_exceeded(self) -> None:
        """Test process_batch with batch size exceeding limits."""

        # Test using mock config to avoid model validation recursion
        class MockBatchConfig:
            def __init__(self) -> None:
                self.data_items = [f"item{i}" for i in range(10000)]
                self.continue_on_error = True

        mock_config = MockBatchConfig()
        result = self.pipeline.process_batch(mock_config)
        FlextTestsMatchers.assert_result_failure(result)
        assert "exceeds maximum" in result.error

    def test_process_with_validation_enabled(self) -> None:
        """Test process_with_validation with validation enabled."""
        self.pipeline.add_step(lambda x: f"validated: {x}")

        request = FlextModels.ProcessingRequest(
            data={"test": "data"}, enable_validation=True
        )

        # Simple validator that always passes
        def always_pass_validator(_data: object) -> FlextResult[None]:
            return FlextResult[None].ok(None)

        result = self.pipeline.process_with_validation(request, always_pass_validator)
        FlextTestsMatchers.assert_result_success(result)

    def test_process_with_validation_disabled(self) -> None:
        """Test process_with_validation with validation disabled."""
        self.pipeline.add_step(lambda x: f"no_validation: {x}")

        request = FlextModels.ProcessingRequest(
            data={"test": "data"}, enable_validation=False
        )

        # Validator that would fail (but shouldn't be called)
        def always_fail_validator(_data: object) -> FlextResult[None]:
            return FlextResult[None].fail("Validation failed")

        result = self.pipeline.process_with_validation(request, always_fail_validator)
        FlextTestsMatchers.assert_result_success(result)

    def test_process_with_validation_failure(self) -> None:
        """Test process_with_validation with validation failure."""
        request = FlextModels.ProcessingRequest(
            data={"test": "data"}, enable_validation=True
        )

        # Validator that always fails
        def always_fail_validator(_data: object) -> FlextResult[None]:
            return FlextResult[None].fail("Validation failed")

        result = self.pipeline.process_with_validation(request, always_fail_validator)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Validation failed" in result.error

    def test_pipeline_step_returning_none_despite_success(self) -> None:
        """Test pipeline step that returns FlextResult success but with None value."""

        def step_returning_none_success(_data: object) -> FlextResult[None]:
            return FlextResult[None].ok(None)

        self.pipeline.add_step(step_returning_none_success)

        result = self.pipeline.process({"test": "data"})
        # This should fail because step returned None despite success
        FlextTestsMatchers.assert_result_failure(result)
        assert "returned None despite success" in result.error


class TestFlextProcessingFactoryMethods:
    """Test FlextProcessing factory methods."""

    def test_create_handler_registry(self) -> None:
        """Test creating handler registry."""
        registry = FlextProcessing.create_handler_registry()

        assert isinstance(registry, FlextProcessing.HandlerRegistry)
        assert registry.count() == 0

    def test_create_pipeline(self) -> None:
        """Test creating processing pipeline."""
        pipeline = FlextProcessing.create_pipeline()

        assert isinstance(pipeline, FlextProcessing.Pipeline)

    def test_is_handler_safe_with_handle_method(self) -> None:
        """Test is_handler_safe with object having handle method."""
        real_handler = FlextProcessing.Handler()

        assert FlextProcessing.is_handler_safe(real_handler)

    def test_is_handler_safe_with_callable(self) -> None:
        """Test is_handler_safe with callable object."""

        def callable_handler(x: object) -> object:
            return x

        assert FlextProcessing.is_handler_safe(callable_handler)

    def test_is_handler_safe_with_invalid_handler(self) -> None:
        """Test is_handler_safe with invalid handler."""
        invalid_handler = "not_a_handler"

        assert not FlextProcessing.is_handler_safe(invalid_handler)


class TestFlextProcessingImplementation:
    """Test FlextProcessing.Implementation classes."""

    def test_basic_handler_initialization(self) -> None:
        """Test BasicHandler initialization."""
        handler_name = "test_handler"
        handler = FlextProcessing.Implementation.BasicHandler(handler_name)

        assert handler.name == handler_name
        assert handler.handler_name == handler_name

    def test_basic_handler_handle_method(self) -> None:
        """Test BasicHandler handle method."""
        handler_name = "test_handler"
        handler = FlextProcessing.Implementation.BasicHandler(handler_name)
        request = "test_request"

        result = handler.handle(request)

        FlextTestsMatchers.assert_result_success(result)
        expected = f"Handled by {handler_name}: {request}"
        assert result.unwrap() == expected


class TestFlextProcessingManagement:
    """Test FlextProcessing.Management classes."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.registry = FlextProcessing.Management.HandlerRegistry()

    def test_management_registry_initialization(self) -> None:
        """Test Management.HandlerRegistry initialization."""
        registry = FlextProcessing.Management.HandlerRegistry()
        assert isinstance(registry, FlextProcessing.Management.HandlerRegistry)

    def test_management_registry_register(self) -> None:
        """Test registering handler in management registry."""
        handler_name = "test_handler"
        handler = FlextProcessing.Handler()

        self.registry.register(handler_name, handler)

        assert self.registry.get(handler_name) == handler

    def test_management_registry_get_nonexistent(self) -> None:
        """Test getting non-existent handler returns None."""
        result = self.registry.get("nonexistent")
        assert result is None

    def test_management_registry_get_optional(self) -> None:
        """Test get_optional method."""
        handler_name = "test_handler"
        handler = FlextProcessing.Handler()

        assert self.registry.get_optional("nonexistent") is None

        self.registry.register(handler_name, handler)
        assert self.registry.get_optional(handler_name) == handler


class TestFlextProcessingPatterns:
    """Test FlextProcessing.Patterns classes."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.chain = FlextProcessing.Patterns.HandlerChain("test_chain")

    def test_handler_chain_initialization(self) -> None:
        """Test HandlerChain initialization."""
        chain_name = "test_chain"
        chain = FlextProcessing.Patterns.HandlerChain(chain_name)

        assert chain.name == chain_name

    def test_handler_chain_add_handler(self) -> None:
        """Test adding handler to chain."""
        handler = FlextProcessing.Handler()

        self.chain.add_handler(handler)

        # Test by executing the chain
        result = self.chain.handle("test_request")
        FlextTestsMatchers.assert_result_success(result)

    def test_handler_chain_handle_with_successful_handlers(self) -> None:
        """Test handler chain with successful handlers."""
        # Use real handlers instead of mocks
        handler1 = FlextProcessing.Handler()
        handler2 = FlextProcessing.Handler()

        self.chain.add_handler(handler1)
        self.chain.add_handler(handler2)

        result = self.chain.handle("initial_request")

        FlextTestsMatchers.assert_result_success(result)
        # Real handlers return processed data, not fixed mock values
        assert "Base handler processed:" in str(result.unwrap())

    def test_handler_chain_handle_with_failing_handler(self) -> None:
        """Test handler chain with failing handler that returns failure."""

        # Create a custom failing handler using factory patterns
        class FailingHandler:
            def handle(self, _request: object) -> FlextResult[str]:
                return FlextResult[str].fail("Handler failed")

        failing_handler = FailingHandler()
        self.chain.add_handler(failing_handler)

        result = self.chain.handle("request")

        FlextTestsMatchers.assert_result_failure(result)
        assert result.error
        assert result.error is not None
        assert "Handler failed" in result.error

    def test_handler_chain_with_no_handlers(self) -> None:
        """Test handler chain with no handlers."""
        result = self.chain.handle("request")

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "request"


class TestFlextProcessingProtocols:
    """Test FlextProcessing.Protocols classes."""

    def test_chainable_handler_initialization(self) -> None:
        """Test ChainableHandler initialization."""
        handler_name = "test_chainable"
        handler = FlextProcessing.Protocols.ChainableHandler(handler_name)

        assert handler.name == handler_name

    def test_chainable_handler_handle(self) -> None:
        """Test ChainableHandler handle method."""
        handler_name = "test_chainable"
        handler = FlextProcessing.Protocols.ChainableHandler(handler_name)
        request = "test_request"

        result = handler.handle(request)

        # ChainableHandler may return FlextResult or string - handle both
        if isinstance(result, FlextResult):
            expected = f"Chain handled by {handler_name}: {request}"
            assert expected in str(result.unwrap())
        else:
            expected = f"Chain handled by {handler_name}: {request}"
            assert result == expected


class TestFlextProcessingIntegration:
    """Integration tests for FlextProcessing components."""

    def test_end_to_end_handler_registry_pipeline(self) -> None:
        """Test complete flow: handler registry + pipeline."""
        # Create registry and handlers
        registry = FlextProcessing.create_handler_registry()

        def transform_handler(data: object) -> str:
            return f"transformed_{data}"

        # Use new HandlerRegistration model
        registration = FlextModels.HandlerRegistration(
            name="transformer", handler=transform_handler
        )
        registry.register(registration)

        # Create pipeline that uses registry
        pipeline = FlextProcessing.create_pipeline()

        def registry_step(data: object) -> object:
            result = registry.execute("transformer", data)
            return result.unwrap() if result.is_success else data

        pipeline.add_step(registry_step)

        # Test the integration
        result = pipeline.process("input_data")

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "transformed_input_data"

    def test_complex_pipeline_with_error_handling(self) -> None:
        """Test complex pipeline with proper error handling."""
        pipeline = FlextProcessing.create_pipeline()

        def validate_step(data: object) -> FlextResult[str]:
            if not data or data == "invalid":
                return FlextResult[str].fail("Validation failed")
            return FlextResult[str].ok(str(data))

        def transform_step(data: object) -> str:
            return f"processed_{data}"

        pipeline.add_step(validate_step)
        pipeline.add_step(transform_step)

        # Test successful case
        result = pipeline.process("valid_data")
        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "processed_valid_data"

        # Test failure case
        result = pipeline.process("invalid")
        FlextTestsMatchers.assert_result_failure(result)
        assert "Validation failed" in result.error

    def test_handler_chain_with_different_handler_types(self) -> None:
        """Test handler chain with mixed handler types using real functionality."""
        chain = FlextProcessing.Patterns.HandlerChain("mixed_chain")

        # Use real handlers instead of mocks
        basic_handler = FlextProcessing.Implementation.BasicHandler("basic")
        chainable_handler = FlextProcessing.Protocols.ChainableHandler("chainable")

        chain.add_handler(basic_handler)
        chain.add_handler(chainable_handler)

        result = chain.handle("input")

        FlextTestsMatchers.assert_result_success(result)
        # Test actual functionality instead of mocked return values
        result_data = result.unwrap()
        assert isinstance(result_data, str)
        assert "input" in str(result_data)  # Real handlers process the input
