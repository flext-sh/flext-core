"""Comprehensive test coverage for FlextProcessing module.

Uses flext_tests patterns for standardized testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Never
from unittest.mock import Mock

from flext_core import FlextProcessing, FlextResult
from flext_tests import FlextTestsFactories, FlextTestsMatchers


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
        self.mock_handler = Mock()
        self.mock_handler.handle = Mock(return_value=FlextResult[str].ok("mock_result"))

    def test_registry_initialization(self) -> None:
        """Test handler registry initialization."""
        registry = FlextProcessing.HandlerRegistry()
        assert registry.count() == 0
        assert not registry.exists("any_handler")

    def test_register_handler_success(self) -> None:
        """Test successful handler registration."""
        handler_name = "test_handler"

        result = self.registry.register(handler_name, self.mock_handler)

        FlextTestsMatchers.assert_result_success(result)
        assert self.registry.count() == 1
        assert self.registry.exists(handler_name)

    def test_register_duplicate_handler_failure(self) -> None:
        """Test registering duplicate handler fails."""
        handler_name = "test_handler"

        # Register first handler
        self.registry.register(handler_name, self.mock_handler)

        # Try to register again
        result = self.registry.register(handler_name, self.mock_handler)

        FlextTestsMatchers.assert_result_failure(result)
        assert f"Handler '{handler_name}' already registered" in str(result.error)

    def test_get_existing_handler_success(self) -> None:
        """Test retrieving existing handler."""
        handler_name = "test_handler"
        self.registry.register(handler_name, self.mock_handler)

        result = self.registry.get(handler_name)

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == self.mock_handler

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
        expected_result = "handled_result"

        self.mock_handler.handle.return_value = FlextResult[str].ok(expected_result)
        self.registry.register(handler_name, self.mock_handler)

        result = self.registry.execute(handler_name, request)

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == expected_result
        self.mock_handler.handle.assert_called_once_with(request)

    def test_execute_callable_handler(self) -> None:
        """Test executing callable handler."""
        handler_name = "callable_handler"
        request = "test_request"
        expected_result = "callable_result"

        def callable_handler(_req: object) -> str:
            return expected_result

        self.registry.register(handler_name, callable_handler)

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

        self.registry.register(handler_name, result_handler)

        result = self.registry.execute(handler_name, request)

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == expected_result

    def test_execute_handler_without_handle_method_or_callable(self) -> None:
        """Test executing handler that's neither callable nor has handle method."""
        handler_name = "invalid_handler"
        invalid_handler = "not_a_handler"

        self.registry.register(handler_name, invalid_handler)

        result = self.registry.execute(handler_name, "request")

        FlextTestsMatchers.assert_result_failure(result)
        assert result.error is not None
        assert (
            f"Handler '{handler_name}' does not implement handle method" in result.error
        )

    def test_execute_handler_with_exception(self) -> None:
        """Test executing handler that raises exception."""
        handler_name = "failing_handler"

        def failing_handler(_req: object) -> Never:
            msg = "Handler failed"
            raise ValueError(msg)

        self.registry.register(handler_name, failing_handler)

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

        self.registry.register("handler1", self.mock_handler)
        assert self.registry.count() == 1

        self.registry.register("handler2", self.mock_handler)
        assert self.registry.count() == 2

    def test_exists_method(self) -> None:
        """Test exists method."""
        handler_name = "test_handler"

        assert not self.registry.exists(handler_name)

        self.registry.register(handler_name, self.mock_handler)
        assert self.registry.exists(handler_name)

    def test_get_optional_method(self) -> None:
        """Test get_optional method."""
        handler_name = "test_handler"

        assert self.registry.get_optional(handler_name) is None

        self.registry.register(handler_name, self.mock_handler)
        assert self.registry.get_optional(handler_name) == self.mock_handler


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
        assert result.error == "Step failed"

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
        mock_handler = Mock()
        mock_handler.handle = Mock()

        assert FlextProcessing.is_handler_safe(mock_handler)

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
        handler = Mock()

        self.registry.register(handler_name, handler)

        assert self.registry.get(handler_name) == handler

    def test_management_registry_get_nonexistent(self) -> None:
        """Test getting non-existent handler returns None."""
        result = self.registry.get("nonexistent")
        assert result is None

    def test_management_registry_get_optional(self) -> None:
        """Test get_optional method."""
        handler_name = "test_handler"
        handler = Mock()

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
        handler = Mock()

        self.chain.add_handler(handler)

        # Test by executing the chain
        result = self.chain.handle("test_request")
        FlextTestsMatchers.assert_result_success(result)

    def test_handler_chain_handle_with_successful_handlers(self) -> None:
        """Test handler chain with successful handlers."""
        handler1 = Mock()
        handler1.handle = Mock(return_value=Mock(success=True, data="result1"))

        handler2 = Mock()
        handler2.handle = Mock(return_value=Mock(success=True, data="result2"))

        self.chain.add_handler(handler1)
        self.chain.add_handler(handler2)

        result = self.chain.handle("initial_request")

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "result2"

    def test_handler_chain_handle_with_failing_handler(self) -> None:
        """Test handler chain with failing handler."""
        failing_handler = Mock()
        failing_handler.handle = Mock(
            return_value=Mock(success=False, error="Handler failed"),
        )

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

        registry.register("transformer", transform_handler)

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
        assert result.error == "Validation failed"

    def test_handler_chain_with_different_handler_types(self) -> None:
        """Test handler chain with mixed handler types."""
        chain = FlextProcessing.Patterns.HandlerChain("mixed_chain")

        # Add BasicHandler with proper attribute assignment
        basic_handler = FlextProcessing.Implementation.BasicHandler("basic")
        # Use setattr to avoid method assignment error
        setattr(
            basic_handler,
            "handle",
            Mock(return_value=Mock(success=True, data="basic_result")),
        )

        # Add ChainableHandler with proper attribute assignment
        chainable_handler = FlextProcessing.Protocols.ChainableHandler("chainable")
        # Use setattr to avoid method assignment error
        setattr(
            chainable_handler,
            "handle",
            Mock(return_value=Mock(success=True, data="chainable_result")),
        )

        chain.add_handler(basic_handler)
        chain.add_handler(chainable_handler)

        result = chain.handle("input")

        FlextTestsMatchers.assert_result_success(result)
        assert result.unwrap() == "chainable_result"
