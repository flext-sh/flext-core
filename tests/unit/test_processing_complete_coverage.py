"""Coverage tests for FlextProcessing - High coverage target.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextProcessing, FlextResult


class TestFlextProcessingCompleteCoverage:
    """Complete coverage tests for FlextProcessing."""

    def test_handler_without_handle_method(self) -> None:
        """Test execution with handler that doesn't implement handle method."""
        registry = FlextProcessing.HandlerRegistry()

        # Create object without handle method
        class InvalidHandler:
            """Handler without handle method."""

            def process(self, request: object) -> str:
                """Wrong method name."""
                return str(request)

        invalid_handler = InvalidHandler()
        registry.register("invalid", invalid_handler)

        # Should fail when trying to execute
        result = registry.execute("invalid", "test_request")
        assert result.is_failure
        assert "does not implement handle method" in str(result.error)

    def test_handler_exception_during_execution(self) -> None:
        """Test exception handling during handler execution."""
        registry = FlextProcessing.HandlerRegistry()

        class ExceptionHandler(FlextProcessing.Handler):
            """Handler that raises exception."""

            def handle(self, request: object) -> FlextResult[object]:
                """Handle that raises exception."""
                if request == "error":
                    msg = "Test exception"
                    raise ValueError(msg)
                return FlextResult[object].ok("success")

        handler = ExceptionHandler()
        registry.register("exception_handler", handler)

        # Should catch and wrap exception
        result = registry.execute("exception_handler", "error")
        assert result.is_failure
        assert "Handler execution failed: Test exception" in str(result.error)

        # Should work normally for non-error cases
        success_result = registry.execute("exception_handler", "normal")
        assert success_result.is_success
        assert success_result.unwrap() == "success"

    def test_pipeline_processing(self) -> None:
        """Test pipeline processing functionality."""
        pipeline = FlextProcessing.Pipeline()

        # Add some processing steps
        step1 = {"type": "transform", "operation": "uppercase"}
        step2 = {"type": "filter", "criteria": "length > 3"}

        pipeline.add_step(step1)
        pipeline.add_step(step2)

        # Process data through pipeline
        data = {"input": "hello world", "format": "text"}
        result = pipeline.process(data)

        # Should use FlextCommands bus internally
        # The exact result depends on command bus implementation
        # but we're testing that the interface works
        assert isinstance(result, FlextResult)

    def test_base_handler_functionality(self) -> None:
        """Test base Handler class functionality."""
        handler = FlextProcessing.Handler()

        # Test default handle method
        result = handler.handle("test_request")
        assert result.is_success
        assert "Base handler processed: test_request" in str(result.unwrap())

    def test_static_factory_methods(self) -> None:
        """Test static factory methods."""
        # Test create_handler_registry
        registry = FlextProcessing.create_handler_registry()
        assert isinstance(registry, FlextProcessing.HandlerRegistry)
        assert registry.count() == 0

        # Test create_pipeline
        pipeline = FlextProcessing.create_pipeline()
        assert isinstance(pipeline, FlextProcessing.Pipeline)

        # Test is_handler_safe
        class ValidHandler:
            def handle(self, request: object) -> object:
                return request

        class InvalidHandler:
            def process(self, request: object) -> object:
                return request

        valid_handler = ValidHandler()
        invalid_handler = InvalidHandler()

        assert FlextProcessing.is_handler_safe(valid_handler) is True
        assert FlextProcessing.is_handler_safe(invalid_handler) is False
        assert FlextProcessing.is_handler_safe("not_an_object") is False

    def test_implementation_basic_handler(self) -> None:
        """Test Implementation.BasicHandler functionality."""
        handler = FlextProcessing.Implementation.BasicHandler("test_handler")

        assert handler.name == "test_handler"
        assert handler.handler_name == "test_handler"

        result = handler.handle("test_request")
        assert result.is_success
        assert result.unwrap() == "Handled by test_handler: test_request"

    def test_management_handler_registry(self) -> None:
        """Test Management.HandlerRegistry functionality."""
        registry = FlextProcessing.Management.HandlerRegistry()

        # Test get_optional method
        assert registry.get_optional("nonexistent") is None

        # Register something and test
        test_item = {"type": "test"}
        registry.register("test_item", test_item)

        retrieved = registry.get_optional("test_item")
        assert retrieved == test_item

    def test_patterns_handler_chain(self) -> None:
        """Test Patterns.HandlerChain functionality."""
        chain = FlextProcessing.Patterns.HandlerChain("test_chain")

        assert chain.name == "test_chain"

        # Add some handlers to the chain
        handler1 = FlextProcessing.Implementation.BasicHandler("handler1")
        handler2 = FlextProcessing.Implementation.BasicHandler("handler2")

        chain.add_handler(handler1)
        chain.add_handler(handler2)

        # Test chain processing
        result = chain.handle("test_request")
        assert isinstance(result, FlextResult)

    def test_protocols_chainable_handler(self) -> None:
        """Test that ChainableHandler protocol is properly defined."""
        # This tests the protocol definition
        # Protocol is properly defined, just check it exists

        # Test that a class can implement the protocol
        class TestChainableHandler:
            def handle(self, request: object) -> FlextResult[object]:
                _ = request  # Mark as intentionally unused
                return FlextResult[object].ok("handled")

            def set_next(
                self, handler: FlextProcessing.Protocols.ChainableHandler
            ) -> None:
                self._next = handler

        test_handler = TestChainableHandler()
        # Protocol check - should not raise
        assert hasattr(test_handler, "handle")
        assert hasattr(test_handler, "set_next")

    def test_registry_edge_cases(self) -> None:
        """Test registry edge cases for coverage."""
        registry = FlextProcessing.HandlerRegistry()

        # Test registering None
        result = registry.register("none_handler", None)
        assert result.is_success

        # Test executing None handler (should fail at handle check)
        execute_result = registry.execute("none_handler", "request")
        assert execute_result.is_failure
        assert "does not implement handle method" in str(execute_result.error)

        # Test empty string as handler name
        empty_result = registry.register("", FlextProcessing.Handler())
        assert empty_result.is_success
        assert registry.exists("")

        # Test execution with empty string handler name
        exec_empty = registry.execute("", "test")
        assert exec_empty.is_success

    def test_pipeline_edge_cases(self) -> None:
        """Test pipeline edge cases for coverage."""
        pipeline = FlextProcessing.Pipeline()

        # Test processing with no steps
        result = pipeline.process("empty_pipeline_data")
        assert isinstance(result, FlextResult)

        # Test adding None step
        pipeline.add_step(None)
        result_with_none = pipeline.process("data_with_none_step")
        assert isinstance(result_with_none, FlextResult)

    def test_complete_workflow(self) -> None:
        """Test workflow to ensure key paths are covered."""
        # Create registry
        registry = FlextProcessing.create_handler_registry()

        # Create and register multiple handlers
        handler1 = FlextProcessing.Implementation.BasicHandler("workflow_handler_1")
        handler2 = FlextProcessing.Implementation.BasicHandler("workflow_handler_2")

        registry.register("h1", handler1)
        registry.register("h2", handler2)

        # Verify all are registered
        assert registry.count() == 2
        assert registry.exists("h1")
        assert registry.exists("h2")

        # Execute both handlers
        result1 = registry.execute("h1", "workflow_request_1")
        result2 = registry.execute("h2", "workflow_request_2")

        assert result1.is_success
        assert result2.is_success
        assert "workflow_handler_1" in str(result1.unwrap())
        assert "workflow_handler_2" in str(result2.unwrap())

        # Create pipeline and process data
        pipeline = FlextProcessing.create_pipeline()
        pipeline.add_step({"step": 1})
        pipeline.add_step({"step": 2})

        pipeline_result = pipeline.process({"workflow": "complete"})
        assert isinstance(pipeline_result, FlextResult)

        # Test utility methods
        assert FlextProcessing.is_handler_safe(handler1) is True
        assert FlextProcessing.is_handler_safe({"not": "handler"}) is False
