"""Comprehensive tests for FLEXT patterns logging module."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

from flext_core.constants import FlextLogLevel
from flext_core.patterns.logging import FlextLogContext
from flext_core.patterns.logging import FlextLogger
from flext_core.patterns.logging import FlextLoggerFactory
from flext_core.patterns.logging import FlextLoggerMixin
from flext_core.patterns.logging import FlextStandardLogger
from flext_core.patterns.logging import create_context_from_dict
from flext_core.patterns.typedefs import FlextLoggerContext
from flext_core.patterns.typedefs import FlextLoggerName
from flext_core.patterns.typedefs import FlextLogTag


class TestFlextLogContext:
    """Test FlextLogContext class."""

    def test_context_creation_with_defaults(self) -> None:
        """Test context creation with default values."""
        context = FlextLogContext()

        assert context.context_id is not None
        assert isinstance(context.context_id, str)
        assert context.context_id.startswith("context_")
        assert context.tags == []
        assert context.metadata == {}

    def test_context_creation_with_values(self) -> None:
        """Test context creation with specific values."""
        context_id = FlextLoggerContext("test_context")
        tags = [FlextLogTag("api"), FlextLogTag("user")]
        metadata = {"user_id": "123", "session": "abc"}

        context = FlextLogContext(
            context_id=context_id,
            tags=tags,
            metadata=metadata,
        )

        assert context.context_id == context_id
        assert context.tags == tags
        assert context.metadata == metadata

    def test_add_tag_new(self) -> None:
        """Test adding a new tag."""
        context = FlextLogContext()
        tag = FlextLogTag("test_tag")

        context.add_tag(tag)

        assert tag in context.tags
        assert len(context.tags) == 1

    def test_add_tag_duplicate(self) -> None:
        """Test adding a duplicate tag."""
        context = FlextLogContext()
        tag = FlextLogTag("test_tag")

        context.add_tag(tag)
        context.add_tag(tag)  # Add same tag again

        assert len(context.tags) == 1
        assert context.tags[0] == tag

    def test_add_metadata(self) -> None:
        """Test adding metadata."""
        context = FlextLogContext()

        context.add_metadata("key1", "value1")
        context.add_metadata("key2", 123)

        assert context.metadata["key1"] == "value1"
        assert context.metadata["key2"] == 123

    def test_merge_contexts(self) -> None:
        """Test merging two contexts."""
        context1 = FlextLogContext(
            tags=[FlextLogTag("tag1")],
            metadata={"key1": "value1"},
        )

        context2 = FlextLogContext(
            tags=[FlextLogTag("tag2"), FlextLogTag("tag1")],  # Duplicate tag
            metadata={"key2": "value2", "key1": "overwritten"},
        )

        context1.merge(context2)

        # Should have both tags but no duplicates
        assert len(context1.tags) == 2
        assert FlextLogTag("tag1") in context1.tags
        assert FlextLogTag("tag2") in context1.tags

        # Metadata should be merged, with second overwriting first
        assert context1.metadata["key1"] == "overwritten"
        assert context1.metadata["key2"] == "value2"

    def test_to_dict(self) -> None:
        """Test converting context to dictionary."""
        context_id = FlextLoggerContext("test_context")
        tags = [FlextLogTag("api"), FlextLogTag("user")]
        metadata = {"user_id": "123"}

        context = FlextLogContext(
            context_id=context_id,
            tags=tags,
            metadata=metadata,
        )

        result = context.to_dict()

        assert result["context_id"] == context_id
        assert result["tags"] == tags
        assert result["metadata"] == metadata


class ConcreteFlextLogger(FlextLogger):
    """Concrete implementation for testing abstract FlextLogger."""

    def __init__(
        self,
        logger_name: FlextLoggerName,
        context: FlextLogContext | None = None,
    ) -> None:
        """Initialize concrete logger."""
        super().__init__(logger_name, context)
        self.logged_messages: list[dict[str, Any]] = []

    def log(
        self,
        level: FlextLogLevel,
        message: str,
        context: FlextLogContext | None = None,
        **kwargs: object,
    ) -> None:
        """Log message to internal list."""
        merged_context = self._merge_context(context)
        self.logged_messages.append(
            {
                "level": level,
                "message": message,
                "context": merged_context,
                "kwargs": kwargs,
            },
        )


class TestFlextLogger:
    """Test FlextLogger abstract base class."""

    def test_logger_creation_minimal(self) -> None:
        """Test logger creation with minimal parameters."""
        logger_name = FlextLoggerName("test_logger")
        logger = ConcreteFlextLogger(logger_name)

        assert logger.logger_name == logger_name
        assert isinstance(logger.default_context, FlextLogContext)

    def test_logger_creation_with_context(self) -> None:
        """Test logger creation with custom context."""
        logger_name = FlextLoggerName("test_logger")
        custom_context = FlextLogContext(
            context_id=FlextLoggerContext("custom"),
            tags=[FlextLogTag("test")],
        )

        logger = ConcreteFlextLogger(logger_name, custom_context)

        assert logger.logger_name == logger_name
        assert logger.default_context is custom_context

    def test_debug_logging(self) -> None:
        """Test debug level logging."""
        logger = ConcreteFlextLogger(FlextLoggerName("test"))

        logger.debug("Debug message", key="value")

        assert len(logger.logged_messages) == 1
        log_entry = logger.logged_messages[0]
        assert log_entry["level"] == FlextLogLevel.DEBUG
        assert log_entry["message"] == "Debug message"
        assert log_entry["kwargs"]["key"] == "value"

    def test_info_logging(self) -> None:
        """Test info level logging."""
        logger = ConcreteFlextLogger(FlextLoggerName("test"))

        logger.info("Info message")

        assert len(logger.logged_messages) == 1
        log_entry = logger.logged_messages[0]
        assert log_entry["level"] == FlextLogLevel.INFO
        assert log_entry["message"] == "Info message"

    def test_warning_logging(self) -> None:
        """Test warning level logging."""
        logger = ConcreteFlextLogger(FlextLoggerName("test"))

        logger.warning("Warning message")

        assert len(logger.logged_messages) == 1
        log_entry = logger.logged_messages[0]
        assert log_entry["level"] == FlextLogLevel.WARNING
        assert log_entry["message"] == "Warning message"

    def test_error_logging_without_exception(self) -> None:
        """Test error level logging without exception."""
        logger = ConcreteFlextLogger(FlextLoggerName("test"))

        logger.error("Error message")

        assert len(logger.logged_messages) == 1
        log_entry = logger.logged_messages[0]
        assert log_entry["level"] == FlextLogLevel.ERROR
        assert log_entry["message"] == "Error message"
        assert "exception" not in log_entry["kwargs"]

    def test_error_logging_with_exception(self) -> None:
        """Test error level logging with exception."""
        logger = ConcreteFlextLogger(FlextLoggerName("test"))
        exception = ValueError("Test error")

        logger.error("Error message", exception=exception)

        assert len(logger.logged_messages) == 1
        log_entry = logger.logged_messages[0]
        assert log_entry["level"] == FlextLogLevel.ERROR
        assert log_entry["message"] == "Error message"

        exception_data = log_entry["kwargs"]["exception"]
        assert exception_data["type"] == "ValueError"
        assert exception_data["message"] == "Test error"
        assert exception_data["args"] == ("Test error",)

    def test_critical_logging_without_exception(self) -> None:
        """Test critical level logging without exception."""
        logger = ConcreteFlextLogger(FlextLoggerName("test"))

        logger.critical("Critical message")

        assert len(logger.logged_messages) == 1
        log_entry = logger.logged_messages[0]
        assert log_entry["level"] == FlextLogLevel.CRITICAL
        assert log_entry["message"] == "Critical message"

    def test_critical_logging_with_exception(self) -> None:
        """Test critical level logging with exception."""
        logger = ConcreteFlextLogger(FlextLoggerName("test"))
        exception = RuntimeError("Critical error")

        logger.critical("Critical message", exception=exception)

        assert len(logger.logged_messages) == 1
        log_entry = logger.logged_messages[0]
        assert log_entry["level"] == FlextLogLevel.CRITICAL

        exception_data = log_entry["kwargs"]["exception"]
        assert exception_data["type"] == "RuntimeError"
        assert exception_data["message"] == "Critical error"

    def test_merge_context_with_none(self) -> None:
        """Test context merging with None context."""
        default_context = FlextLogContext(
            tags=[FlextLogTag("default")],
            metadata={"default_key": "default_value"},
        )
        logger = ConcreteFlextLogger(FlextLoggerName("test"), default_context)

        merged = logger._merge_context(None)

        assert merged.context_id == default_context.context_id
        assert FlextLogTag("default") in merged.tags
        assert merged.metadata["default_key"] == "default_value"

    def test_merge_context_with_custom(self) -> None:
        """Test context merging with custom context."""
        default_context = FlextLogContext(
            tags=[FlextLogTag("default")],
            metadata={"default_key": "default_value"},
        )
        logger = ConcreteFlextLogger(FlextLoggerName("test"), default_context)

        custom_context = FlextLogContext(
            tags=[FlextLogTag("custom")],
            metadata={"custom_key": "custom_value"},
        )

        merged = logger._merge_context(custom_context)

        # Should have both tags
        assert FlextLogTag("default") in merged.tags
        assert FlextLogTag("custom") in merged.tags

        # Should have both metadata
        assert merged.metadata["default_key"] == "default_value"
        assert merged.metadata["custom_key"] == "custom_value"


class TestFlextStandardLogger:
    """Test FlextStandardLogger concrete implementation."""

    def test_standard_logger_creation_minimal(self) -> None:
        """Test standard logger creation with minimal parameters."""
        logger_name = FlextLoggerName("test_logger")
        logger = FlextStandardLogger(logger_name)

        assert logger.logger_name == logger_name
        assert isinstance(logger.python_logger, logging.Logger)
        assert logger.python_logger.name == str(logger_name)

    def test_standard_logger_creation_with_custom_python_logger(self) -> None:
        """Test standard logger creation with custom Python logger."""
        logger_name = FlextLoggerName("test_logger")
        custom_logger = logging.getLogger("custom")

        logger = FlextStandardLogger(logger_name, python_logger=custom_logger)

        assert logger.python_logger is custom_logger

    @patch("logging.Logger.log")
    def test_log_method_calls_python_logger(self, mock_log: MagicMock) -> None:
        """Test that log method calls Python logger correctly."""
        logger = FlextStandardLogger(FlextLoggerName("test"))
        context = FlextLogContext(tags=[FlextLogTag("test")])

        logger.log(
            FlextLogLevel.INFO,
            "Test message",
            context,
            extra_key="value",
        )

        # Verify Python logger was called
        mock_log.assert_called_once()
        call_args = mock_log.call_args

        # Check logging level
        assert call_args[0][0] == logging.INFO

        # Check message format
        message = call_args[0][1]
        assert "Test message" in message
        assert "test" in message  # Tag should be in message

        # Check extra data
        extra_data = call_args[1]["extra"]["flext_data"]
        assert extra_data["message"] == "Test message"
        assert extra_data["extra_key"] == "value"

    def test_convert_log_level_all_levels(self) -> None:
        """Test conversion of all FlextLogLevel values."""
        logger = FlextStandardLogger(FlextLoggerName("test"))

        assert (
            logger._convert_log_level(FlextLogLevel.CRITICAL)
            == logging.CRITICAL
        )
        assert logger._convert_log_level(FlextLogLevel.ERROR) == logging.ERROR
        assert (
            logger._convert_log_level(FlextLogLevel.WARNING) == logging.WARNING
        )
        assert logger._convert_log_level(FlextLogLevel.INFO) == logging.INFO
        assert logger._convert_log_level(FlextLogLevel.DEBUG) == logging.DEBUG
        assert logger._convert_log_level(FlextLogLevel.TRACE) == logging.DEBUG

    def test_convert_log_level_unknown(self) -> None:
        """Test converting unknown log level."""
        logger = FlextStandardLogger(FlextLoggerName("test"))

        # Create a mock unknown level
        unknown_level = "UNKNOWN"
        result = logger._convert_log_level(unknown_level)

        assert result == logging.INFO  # Should default to INFO

    def test_format_message_with_context_and_tags(self) -> None:
        """Test message formatting with context and tags."""
        logger = FlextStandardLogger(FlextLoggerName("test"))

        log_data = {
            "context": {
                "context_id": "test_context",
                "tags": ["api", "user"],
                "metadata": {},
            },
        }

        result = logger._format_message("Test message", log_data)

        assert result == "[test_context][api,user] Test message"

    def test_format_message_without_tags(self) -> None:
        """Test message formatting without tags."""
        logger = FlextStandardLogger(FlextLoggerName("test"))

        log_data = {
            "context": {
                "context_id": "test_context",
                "tags": [],
                "metadata": {},
            },
        }

        result = logger._format_message("Test message", log_data)

        assert result == "[test_context] Test message"

    def test_format_message_empty_context(self) -> None:
        """Test message formatting with empty context."""
        logger = FlextStandardLogger(FlextLoggerName("test"))

        log_data: dict[str, Any] = {"context": {}}

        result = logger._format_message("Test message", log_data)

        assert result == "[] Test message"

    def test_format_message_with_tags(self) -> None:
        """Test message formatting with tags."""
        logger = FlextStandardLogger(FlextLoggerName("test"))

        log_data = {
            "context": {
                "context_id": "test_context",
                "tags": ["tag1", "tag2"],
                "metadata": {},
            },
        }

        message = logger._format_message("Test message", log_data)
        assert "tag1" in message
        assert "tag2" in message
        assert "test_context" in message


class TestFlextLoggerFactory:
    """Test FlextLoggerFactory class."""

    def test_factory_creation_with_defaults(self) -> None:
        """Test factory creation with default values."""
        factory = FlextLoggerFactory()

        assert factory.default_level == FlextLogLevel.INFO
        assert isinstance(factory.default_context, FlextLogContext)
        assert factory._loggers == {}

    def test_factory_creation_with_custom_values(self) -> None:
        """Test factory creation with custom values."""
        custom_context = FlextLogContext(tags=[FlextLogTag("factory")])
        factory = FlextLoggerFactory(
            default_level=FlextLogLevel.DEBUG,
            default_context=custom_context,
        )

        assert factory.default_level == FlextLogLevel.DEBUG
        assert factory.default_context is custom_context

    def test_create_logger_new(self) -> None:
        """Test creating a new logger."""
        factory = FlextLoggerFactory()
        logger_name = FlextLoggerName("test_logger")

        logger = factory.create_logger(logger_name)

        assert isinstance(logger, FlextStandardLogger)
        assert logger.logger_name == logger_name
        assert logger_name in factory._loggers

    def test_create_logger_existing(self) -> None:
        """Test getting an existing logger."""
        factory = FlextLoggerFactory()
        logger_name = FlextLoggerName("test_logger")

        logger1 = factory.create_logger(logger_name)
        logger2 = factory.create_logger(logger_name)

        assert logger1 is logger2  # Should return same instance

    def test_create_logger_with_custom_context(self) -> None:
        """Test creating logger with custom context."""
        factory_context = FlextLogContext(tags=[FlextLogTag("factory")])
        factory = FlextLoggerFactory(default_context=factory_context)

        custom_context = FlextLogContext(tags=[FlextLogTag("custom")])
        logger_name = FlextLoggerName("test")

        logger = factory.create_logger(logger_name, custom_context)

        # Logger context should have merged tags
        assert FlextLogTag("factory") in logger.default_context.tags
        assert FlextLogTag("custom") in logger.default_context.tags

    def test_create_logger_with_custom_type(self) -> None:
        """Test creating logger with custom logger type."""
        factory = FlextLoggerFactory()
        logger_name = FlextLoggerName("test")

        logger = factory.create_logger(
            logger_name,
            logger_type=ConcreteFlextLogger,
        )

        assert isinstance(logger, ConcreteFlextLogger)

    def test_get_logger_existing(self) -> None:
        """Test getting an existing logger."""
        factory = FlextLoggerFactory()
        logger_name = FlextLoggerName("test")

        created_logger = factory.create_logger(logger_name)
        retrieved_logger = factory.get_logger(logger_name)

        assert retrieved_logger is created_logger

    def test_get_logger_non_existent(self) -> None:
        """Test getting a non-existent logger."""
        factory = FlextLoggerFactory()

        result = factory.get_logger(FlextLoggerName("non_existent"))

        assert result is None

    @patch("logging.basicConfig")
    def test_configure_python_logging_default(
        self,
        mock_basic_config: MagicMock,
    ) -> None:
        """Test configuring Python logging with defaults."""
        factory = FlextLoggerFactory()

        factory.configure_python_logging()

        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == logging.INFO
        assert "%(asctime)s" in call_kwargs["format"]
        assert call_kwargs["datefmt"] == "%Y-%m-%d %H:%M:%S"

    @patch("logging.basicConfig")
    def test_configure_python_logging_custom(
        self,
        mock_basic_config: MagicMock,
    ) -> None:
        """Test configuring Python logging with custom values."""
        factory = FlextLoggerFactory()
        custom_format = "%(levelname)s: %(message)s"

        factory.configure_python_logging(
            level=FlextLogLevel.DEBUG,
            format_string=custom_format,
        )

        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == logging.DEBUG
        assert call_kwargs["format"] == custom_format

    def test_get_all_loggers_empty(self) -> None:
        """Test getting all loggers when none exist."""
        factory = FlextLoggerFactory()

        result = factory.get_all_loggers()

        assert result == {}

    def test_get_all_loggers_with_loggers(self) -> None:
        """Test getting all loggers when some exist."""
        factory = FlextLoggerFactory()

        logger1 = factory.create_logger(FlextLoggerName("logger1"))
        logger2 = factory.create_logger(FlextLoggerName("logger2"))

        result = factory.get_all_loggers()

        assert len(result) == 2
        assert result["logger1"] is logger1
        assert result["logger2"] is logger2

        # Should return a copy, not the original dict
        assert result is not factory._loggers

    def test_access_loggers_private(self) -> None:
        """Test access to private _loggers attribute."""
        factory = FlextLoggerFactory()
        _ = factory._loggers

    def test_access_loggers_private_in_logger(self) -> None:
        """Test access to private _loggers attribute in logger."""
        logger = FlextLoggerFactory()
        _ = logger._loggers


class TestFlextLoggerMixin:
    """Test FlextLoggerMixin class."""

    def test_mixin_initialization(self) -> None:
        """Test mixin initialization creates logger."""

        class TestClass(FlextLoggerMixin):
            """Test class using the mixin."""

            def __init__(self) -> None:
                """Initialize test class."""
                super().__init__()

        instance = TestClass()

        assert hasattr(instance, "_logger")
        assert isinstance(instance.logger, FlextLogger)
        assert instance.logger.logger_name == "TestClass"

        # Should have class-specific tags
        assert FlextLogTag("class") in instance.logger.default_context.tags
        assert FlextLogTag("TestClass") in instance.logger.default_context.tags

    def test_mixin_with_inheritance(self) -> None:
        """Test mixin works with inheritance chain."""

        class BaseClass:
            """Base class."""

            def __init__(self, value: str) -> None:
                """Initialize base class."""
                self.value = value

        class TestClass(FlextLoggerMixin, BaseClass):
            """Test class with multiple inheritance."""

            def __init__(self, value: str) -> None:
                """Initialize test class."""
                super().__init__(value=value)

        instance = TestClass("test_value")

        assert instance.value == "test_value"
        assert isinstance(instance.logger, FlextLogger)
        assert FlextLogTag("TestClass") in instance.logger.default_context.tags


class TestCreateContextFromDict:
    """Test create_context_from_dict utility function."""

    def test_create_context_empty_dict(self) -> None:
        """Test creating context from empty dictionary."""
        result = create_context_from_dict({})

        assert isinstance(result, FlextLogContext)
        # FlextLogContext creates a default context_id when None is passed
        assert result.context_id is not None
        assert isinstance(result.context_id, str)
        assert result.context_id.startswith("context_")
        assert result.tags == []
        assert result.metadata == {}

    def test_create_context_with_valid_data(self) -> None:
        """Test creating context with valid data."""
        data = {
            "context_id": "test_context",
            "tags": ["api", "user", "auth"],
            "metadata": {"user_id": "123", "session": "abc"},
        }

        result = create_context_from_dict(data)

        assert result.context_id == "test_context"
        assert len(result.tags) == 3
        assert FlextLogTag("api") in result.tags
        assert FlextLogTag("user") in result.tags
        assert FlextLogTag("auth") in result.tags
        assert result.metadata == data["metadata"]

    def test_create_context_with_invalid_context_id(self) -> None:
        """Test creating context with invalid context_id."""
        data = {"context_id": 123}  # Invalid type

        result = create_context_from_dict(data)

        # FlextLogContext creates default context_id when None is passed
        assert result.context_id is not None
        assert isinstance(result.context_id, str)
        assert result.context_id.startswith("context_")

    def test_create_context_with_invalid_tags(self) -> None:
        """Test creating context with invalid tags."""
        data = {
            "tags": ["valid", 123, None, "also_valid"],  # Mixed types
        }

        result = create_context_from_dict(data)

        assert len(result.tags) == 2
        assert FlextLogTag("valid") in result.tags
        assert FlextLogTag("also_valid") in result.tags

    def test_create_context_with_non_list_tags(self) -> None:
        """Test creating context with non-list tags."""
        data = {"tags": "not_a_list"}

        result = create_context_from_dict(data)

        assert result.tags == []

    def test_create_context_with_invalid_metadata(self) -> None:
        """Test creating context with invalid metadata."""
        data = {"metadata": "not_a_dict"}

        result = create_context_from_dict(data)

        assert result.metadata == {}

    def test_create_context_with_mixed_valid_invalid(self) -> None:
        """Test creating context with mix of valid and invalid data."""
        data = {
            "context_id": "valid_context",
            "tags": ["valid_tag", 123, "another_valid"],
            "metadata": {"key": "value"},
            "extra_field": "ignored",
        }

        result = create_context_from_dict(data)

        assert result.context_id == "valid_context"
        assert len(result.tags) == 2
        assert FlextLogTag("valid_tag") in result.tags
        assert FlextLogTag("another_valid") in result.tags
        assert result.metadata == {"key": "value"}
