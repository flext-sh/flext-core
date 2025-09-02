"""Test suite for logging_new module."""

# # from pydantic import BaseModel  # Using FlextModels.BaseConfig instead
from flext_core import FlextModels
from flext_core.logging_new import FlextLoggingCore, FlextLoggingMixin
from flext_core.loggings import FlextLogger


class TestModel(FlextModels.BaseConfig):
    """Test Pydantic model."""

    name: str
    value: int


class DummyObject:
    """Dummy object for testing that supports dynamic attributes."""

    def __setattr__(self, name: str, value: object) -> None:
        """Set attribute dynamically."""
        object.__setattr__(self, name, value)

    def __getattribute__(self, name: str) -> object:
        """Get attribute dynamically."""
        return object.__getattribute__(self, name)


class TestFlextLoggingCore:
    """Test suite for FlextLoggingCore."""

    def test_get_logger(self) -> None:
        """Test getting logger for an object."""
        obj = DummyObject()
        logger = FlextLoggingCore.get_logger(obj)
        assert isinstance(logger, FlextLogger)

        # Should return same logger for same object
        logger2 = FlextLoggingCore.get_logger(obj)
        assert logger is logger2

    def test_log_operation(self) -> None:
        """Test logging an operation."""
        obj = DummyObject()

        # Should not raise
        FlextLoggingCore.log_operation(
            obj, "test_operation", user="john", action="test"
        )

    def test_log_error(self) -> None:
        """Test error logging."""
        obj = DummyObject()

        # Test with string error
        FlextLoggingCore.log_error(obj, "Test error", context="testing")

        # Test with exception
        error = ValueError("Test exception")
        FlextLoggingCore.log_error(obj, error, context="exception_test")

    def test_log_info(self) -> None:
        """Test info logging."""
        obj = DummyObject()
        FlextLoggingCore.log_info(obj, "Info message", data="test")

    def test_log_debug(self) -> None:
        """Test debug logging."""
        obj = DummyObject()
        FlextLoggingCore.log_debug(obj, "Debug message", verbose=True)

    def test_log_warning(self) -> None:
        """Test warning logging."""
        obj = DummyObject()
        FlextLoggingCore.log_warning(obj, "Warning message", severity="low")

    def test_log_critical(self) -> None:
        """Test critical logging."""
        obj = DummyObject()
        FlextLoggingCore.log_critical(obj, "Critical error", immediate=True)

    def test_log_exception(self) -> None:
        """Test exception logging."""
        obj = DummyObject()
        FlextLoggingCore.log_exception(obj, "Exception occurred", traceback="...")

    def test_with_logger_context(self) -> None:
        """Test creating logger with context."""
        obj = DummyObject()
        logger = FlextLoggingCore.with_logger_context(
            obj, request_id="123", user="john"
        )
        assert isinstance(logger, FlextLogger)

    def test_start_operation(self) -> None:
        """Test starting an operation."""
        obj = DummyObject()
        operation_id = FlextLoggingCore.start_operation(
            obj, "data_processing", input_size=100
        )
        assert isinstance(operation_id, str)
        assert len(operation_id) > 0

    def test_complete_operation(self) -> None:
        """Test completing an operation."""
        obj = DummyObject()

        # Start an operation
        operation_id = FlextLoggingCore.start_operation(obj, "test_op")

        # Complete it successfully
        FlextLoggingCore.complete_operation(
            obj, operation_id, success=True, result="completed"
        )

        # Complete with failure
        operation_id2 = FlextLoggingCore.start_operation(obj, "failing_op")
        FlextLoggingCore.complete_operation(
            obj, operation_id2, success=False, error="failed"
        )

    def test_logging_with_pydantic_model(self) -> None:
        """Test logging with Pydantic model in context."""
        obj = DummyObject()
        model = TestModel(name="test", value=42)

        FlextLoggingCore.log_info(obj, "Model test", model=model, extra="data")

    def test_logging_with_various_types(self) -> None:
        """Test logging with various data types."""
        obj = DummyObject()

        FlextLoggingCore.log_operation(
            obj,
            "type_test",
            string="text",
            number=123,
            boolean=True,
            none_value=None,
            list_data=[1, 2, 3],
            dict_data={"key": "value"},
            model=TestModel(name="model", value=99),
        )


class TestFlextLoggingMixin:
    """Test suite for FlextLoggingMixin."""

    def test_mixin_provides_logging_methods(self) -> None:
        """Test that mixin provides logging methods."""

        class TestClass(FlextLoggingMixin):
            """Test class using logging mixin."""

        obj = TestClass()

        # Check that logging methods are available
        assert hasattr(obj, "log_operation")
        assert hasattr(obj, "log_error")
        assert hasattr(obj, "log_info")
        assert hasattr(obj, "log_debug")
        assert hasattr(obj, "log_warning")
        assert hasattr(obj, "log_critical")
        assert hasattr(obj, "log_exception")

    def test_mixin_logging_operations(self) -> None:
        """Test mixin logging operations."""

        class TestClass(FlextLoggingMixin):
            """Test class using logging mixin."""

            def perform_action(self) -> None:
                """Perform an action with logging."""
                self.log_info("Starting action")
                self.log_operation("action", step="processing")
                self.log_debug("Debug info", details="verbose")
                self.log_info("Action completed")

        obj = TestClass()

        # Should not raise
        obj.perform_action()

    def test_mixin_error_handling(self) -> None:
        """Test mixin error handling."""

        class TestClass(FlextLoggingMixin):
            """Test class using logging mixin."""

            def handle_error(self) -> None:
                """Handle an error with logging."""
                try:
                    msg = "Test error"
                    raise ValueError(msg)
                except ValueError as e:
                    self.log_error(e, context="error_handling")
                    self.log_exception("Exception caught")

        obj = TestClass()

        # Should not raise
        obj.handle_error()

    def test_mixin_operation_tracking(self) -> None:
        """Test mixin operation tracking."""

        class TestClass(FlextLoggingMixin):
            """Test class using logging mixin."""

            def tracked_operation(self) -> None:
                """Perform a tracked operation."""
                op_id = self.start_operation("tracked_op", size=100)
                try:
                    # Simulate work
                    self.log_info("Processing", operation_id=op_id)
                    self.complete_operation(op_id, success=True)
                except Exception as e:
                    self.complete_operation(op_id, success=False, error=str(e))
                    raise

        obj = TestClass()

        # Should not raise
        obj.tracked_operation()

    def test_mixin_with_context(self) -> None:
        """Test mixin with logger context."""

        class TestClass(FlextLoggingMixin):
            """Test class using logging mixin."""

            def with_context_operation(self) -> None:
                """Perform operation with context."""
                logger = self.with_logger_context(request_id="req123", user="test_user")
                # Logger should have context bound
                assert isinstance(logger, FlextLogger)

        obj = TestClass()
        obj.with_context_operation()
