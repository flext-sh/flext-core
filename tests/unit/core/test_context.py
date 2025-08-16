"""Tests for FlextContext context management system.

Tests context management functionality including correlation IDs,
service identification, request metadata, and serialization.
"""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Thread
from time import sleep

import pytest

from flext_core import FlextContext

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestCorrelationIdManagement:
    """Test correlation ID functionality."""

    def test_get_correlation_id_default(self) -> None:
        """Test getting correlation ID returns None by default."""
        # Clear any existing context
        FlextContext.clear_context()

        assert FlextContext.get_correlation_id() is None

    def test_set_get_correlation_id(self) -> None:
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-123"
        FlextContext.set_correlation_id(test_id)

        assert FlextContext.get_correlation_id() == test_id

    def test_generate_correlation_id(self) -> None:
        """Test generating new correlation ID."""
        generated_id = FlextContext.generate_correlation_id()

        assert generated_id is not None
        assert len(generated_id) == 36  # UUID4 format
        assert FlextContext.get_correlation_id() == generated_id

    def test_parent_correlation_id(self) -> None:
        """Test parent correlation ID functionality."""
        FlextContext.clear_context()

        # Test default is None
        assert FlextContext.get_parent_correlation_id() is None

        # Test setting and getting
        parent_id = "parent-correlation-456"
        FlextContext.set_parent_correlation_id(parent_id)
        assert FlextContext.get_parent_correlation_id() == parent_id

    def test_new_correlation_context_with_id(self) -> None:
        """Test new correlation context with specific ID."""
        test_id = "specific-correlation-789"

        with FlextContext.new_correlation(correlation_id=test_id) as context_id:
            assert context_id == test_id
            assert FlextContext.get_correlation_id() == test_id

    def test_new_correlation_context_generated(self) -> None:
        """Test new correlation context with generated ID."""
        with FlextContext.new_correlation() as context_id:
            assert context_id is not None
            assert len(context_id) == 36  # UUID4 format
            assert FlextContext.get_correlation_id() == context_id

    def test_new_correlation_with_parent(self) -> None:
        """Test new correlation context with parent ID."""
        parent_id = "explicit-parent-123"

        with FlextContext.new_correlation(parent_id=parent_id) as context_id:
            assert context_id is not None
            assert FlextContext.get_parent_correlation_id() == parent_id

    def test_new_correlation_inherits_current_as_parent(self) -> None:
        """Test that new correlation inherits current as parent."""
        # Set initial correlation
        initial_id = "initial-correlation"
        FlextContext.set_correlation_id(initial_id)

        with FlextContext.new_correlation() as new_id:
            assert new_id != initial_id
            assert FlextContext.get_parent_correlation_id() == initial_id

    def test_new_correlation_context_restoration(self) -> None:
        """Test context restoration after new_correlation."""
        # Set initial context
        initial_correlation = "initial-123"
        initial_parent = "initial-parent-456"
        FlextContext.set_correlation_id(initial_correlation)
        FlextContext.set_parent_correlation_id(initial_parent)

        # Create new context
        with FlextContext.new_correlation(correlation_id="temp-789"):
            assert FlextContext.get_correlation_id() == "temp-789"

        # Context should be restored (correlation cleared, parent restored)
        assert FlextContext.get_correlation_id() == initial_correlation

    def test_inherit_correlation_existing(self) -> None:
        """Test inheriting existing correlation ID."""
        existing_id = "existing-correlation"
        FlextContext.set_correlation_id(existing_id)

        with FlextContext.inherit_correlation() as inherited_id:
            assert inherited_id == existing_id
            assert FlextContext.get_correlation_id() == existing_id

    def test_inherit_correlation_creates_new(self) -> None:
        """Test inherit_correlation creates new when none exists."""
        FlextContext.clear_context()

        with FlextContext.inherit_correlation() as inherited_id:
            assert inherited_id is not None
            assert len(inherited_id) == 36  # UUID4 format
            assert FlextContext.get_correlation_id() == inherited_id


class TestServiceIdentification:
    """Test service identification functionality."""

    def test_service_name_default(self) -> None:
        """Test service name is None by default."""
        FlextContext.clear_context()
        assert FlextContext.get_service_name() is None

    def test_set_get_service_name(self) -> None:
        """Test setting and getting service name."""
        service_name = "test-service"
        FlextContext.set_service_name(service_name)
        assert FlextContext.get_service_name() == service_name

    def test_service_version_default(self) -> None:
        """Test service version is None by default."""
        FlextContext.clear_context()
        assert FlextContext.get_service_version() is None

    def test_set_get_service_version(self) -> None:
        """Test setting and getting service version."""
        version = "1.2.3"
        FlextContext.set_service_version(version)
        assert FlextContext.get_service_version() == version

    def test_service_context_both_values(self) -> None:
        """Test service context with name and version."""
        service_name = "test-service"
        version = "2.0.0"

        with FlextContext.service_context(service_name, version):
            assert FlextContext.get_service_name() == service_name
            assert FlextContext.get_service_version() == version

    def test_service_context_name_only(self) -> None:
        """Test service context with name only."""
        FlextContext.clear_context()  # Clear any state from previous tests
        service_name = "name-only-service"

        with FlextContext.service_context(service_name):
            assert FlextContext.get_service_name() == service_name
            assert FlextContext.get_service_version() is None

    def test_service_context_restoration(self) -> None:
        """Test service context restoration after block."""
        # Set initial values
        initial_name = "initial-service"
        initial_version = "1.0.0"
        FlextContext.set_service_name(initial_name)
        FlextContext.set_service_version(initial_version)

        # Use service context
        with FlextContext.service_context("temp-service", "2.0.0"):
            assert FlextContext.get_service_name() == "temp-service"
            assert FlextContext.get_service_version() == "2.0.0"

        # Should restore initial values
        assert FlextContext.get_service_name() == initial_name
        assert FlextContext.get_service_version() == initial_version


class TestRequestMetadata:
    """Test request metadata functionality."""

    def test_user_id_default(self) -> None:
        """Test user ID is None by default."""
        FlextContext.clear_context()
        assert FlextContext.get_user_id() is None

    def test_set_get_user_id(self) -> None:
        """Test setting and getting user ID."""
        user_id = "user-123"
        FlextContext.set_user_id(user_id)
        assert FlextContext.get_user_id() == user_id

    def test_operation_name_default(self) -> None:
        """Test operation name is None by default."""
        FlextContext.clear_context()
        assert FlextContext.get_operation_name() is None

    def test_set_get_operation_name(self) -> None:
        """Test setting and getting operation name."""
        operation = "create_user"
        FlextContext.set_operation_name(operation)
        assert FlextContext.get_operation_name() == operation

    def test_request_id_default(self) -> None:
        """Test request ID is None by default."""
        FlextContext.clear_context()
        assert FlextContext.get_request_id() is None

    def test_set_get_request_id(self) -> None:
        """Test setting and getting request ID."""
        request_id = "request-456"
        FlextContext.set_request_id(request_id)
        assert FlextContext.get_request_id() == request_id

    def test_request_context_all_params(self) -> None:
        """Test request context with all parameters."""
        user_id = "test-user"
        operation_name = "test-operation"
        request_id = "test-request"
        metadata = {"key": "value", "num": 123}

        with FlextContext.request_context(
            user_id=user_id,
            operation_name=operation_name,
            request_id=request_id,
            metadata=metadata,
        ):
            assert FlextContext.get_user_id() == user_id
            assert FlextContext.get_operation_name() == operation_name
            assert FlextContext.get_request_id() == request_id
            assert FlextContext.get_operation_metadata() == metadata

    def test_request_context_partial_params(self) -> None:
        """Test request context with some parameters."""
        # Clear context to avoid contamination from other tests
        FlextContext.clear_context()

        user_id = "partial-user"

        with FlextContext.request_context(user_id=user_id):
            assert FlextContext.get_user_id() == user_id
            assert FlextContext.get_operation_name() is None
            assert FlextContext.get_request_id() is None

    def test_request_context_restoration(self) -> None:
        """Test request context restoration."""
        # Set initial values
        FlextContext.set_user_id("initial-user")
        FlextContext.set_operation_name("initial-op")

        with FlextContext.request_context(
            user_id="temp-user",
            operation_name="temp-op",
        ):
            assert FlextContext.get_user_id() == "temp-user"
            assert FlextContext.get_operation_name() == "temp-op"

        # Should restore initial values
        assert FlextContext.get_user_id() == "initial-user"
        assert FlextContext.get_operation_name() == "initial-op"


class TestPerformanceContext:
    """Test performance context functionality."""

    def test_operation_start_time_default(self) -> None:
        """Test operation start time is None by default."""
        FlextContext.clear_context()
        assert FlextContext.get_operation_start_time() is None

    def test_set_operation_start_time_explicit(self) -> None:
        """Test setting explicit start time."""
        start_time = datetime.now(UTC)
        FlextContext.set_operation_start_time(start_time)
        assert FlextContext.get_operation_start_time() == start_time

    def test_set_operation_start_time_auto(self) -> None:
        """Test setting start time automatically."""
        FlextContext.set_operation_start_time()
        start_time = FlextContext.get_operation_start_time()

        assert start_time is not None
        assert isinstance(start_time, datetime)
        assert start_time.tzinfo == UTC

    def test_operation_metadata_default(self) -> None:
        """Test operation metadata is None by default."""
        FlextContext.clear_context()
        assert FlextContext.get_operation_metadata() is None

    def test_set_get_operation_metadata(self) -> None:
        """Test setting and getting operation metadata."""
        metadata = {"step": "validation", "count": 10}
        FlextContext.set_operation_metadata(metadata)
        assert FlextContext.get_operation_metadata() == metadata

    def test_add_operation_metadata_new(self) -> None:
        """Test adding metadata when none exists."""
        FlextContext.clear_context()
        FlextContext.add_operation_metadata("key1", "value1")

        metadata = FlextContext.get_operation_metadata()
        assert metadata == {"key1": "value1"}

    def test_add_operation_metadata_existing(self) -> None:
        """Test adding to existing metadata."""
        initial: dict[str, object] = {"existing": "value"}
        FlextContext.set_operation_metadata(initial)

        FlextContext.add_operation_metadata("new_key", "new_value")

        metadata = FlextContext.get_operation_metadata()
        assert metadata == {"existing": "value", "new_key": "new_value"}

    def test_timed_operation_basic(self) -> None:
        """Test basic timed operation."""
        operation_name = "test_operation"

        with FlextContext.timed_operation(operation_name) as metadata:
            # Check initial metadata
            assert "start_time" in metadata
            assert "operation_name" in metadata
            assert metadata["operation_name"] == operation_name

            # Verify context is set
            assert FlextContext.get_operation_name() == operation_name
            assert FlextContext.get_operation_start_time() is not None

            # Small delay to measure duration
            sleep(0.01)

        # Check final metadata has duration
        assert "end_time" in metadata
        assert "duration_seconds" in metadata
        assert isinstance(metadata["duration_seconds"], float)
        assert metadata["duration_seconds"] > 0

    def test_timed_operation_no_name(self) -> None:
        """Test timed operation without name."""
        with FlextContext.timed_operation() as metadata:
            assert "start_time" in metadata
            assert metadata["operation_name"] is None

            sleep(0.001)  # Very small delay

        assert "duration_seconds" in metadata
        duration = metadata["duration_seconds"]
        assert isinstance(duration, (int, float))
        assert duration >= 0


class TestContextSerialization:
    """Test context serialization functionality."""

    def test_get_full_context_empty(self) -> None:
        """Test getting full context when empty."""
        FlextContext.clear_context()
        context = FlextContext.get_full_context()

        expected_keys = {
            "correlation_id",
            "parent_correlation_id",
            "service_name",
            "service_version",
            "user_id",
            "operation_name",
            "request_id",
            "operation_start_time",
            "operation_metadata",
        }

        assert set(context.keys()) == expected_keys
        assert all(value is None for value in context.values())

    def test_get_full_context_populated(self) -> None:
        """Test getting full context when populated."""
        # Set various context values
        FlextContext.set_correlation_id("corr-123")
        FlextContext.set_parent_correlation_id("parent-456")
        FlextContext.set_service_name("test-service")
        FlextContext.set_service_version("1.0.0")
        FlextContext.set_user_id("user-789")
        FlextContext.set_operation_name("test-op")
        FlextContext.set_request_id("req-012")

        start_time = datetime.now(UTC)
        metadata: dict[str, object] = {"key": "value"}
        FlextContext.set_operation_start_time(start_time)
        FlextContext.set_operation_metadata(metadata)

        context = FlextContext.get_full_context()

        assert context["correlation_id"] == "corr-123"
        assert context["parent_correlation_id"] == "parent-456"
        assert context["service_name"] == "test-service"
        assert context["service_version"] == "1.0.0"
        assert context["user_id"] == "user-789"
        assert context["operation_name"] == "test-op"
        assert context["request_id"] == "req-012"
        assert context["operation_start_time"] == start_time
        assert context["operation_metadata"] == metadata

    def test_get_correlation_context_empty(self) -> None:
        """Test getting correlation context when empty."""
        FlextContext.clear_context()
        context = FlextContext.get_correlation_context()

        assert context == {}

    def test_get_correlation_context_populated(self) -> None:
        """Test getting correlation context with data."""
        FlextContext.set_correlation_id("corr-123")
        FlextContext.set_parent_correlation_id("parent-456")
        FlextContext.set_service_name("test-service")

        context = FlextContext.get_correlation_context()

        assert context["X-Correlation-Id"] == "corr-123"
        assert context["X-Parent-Correlation-Id"] == "parent-456"
        assert context["X-Service-Name"] == "test-service"

    def test_set_from_context_http_headers(self) -> None:
        """Test setting context from HTTP header format."""
        headers = {
            "X-Correlation-Id": "header-corr-123",
            "X-Parent-Correlation-Id": "header-parent-456",
            "X-Service-Name": "header-service",
            "X-User-Id": "header-user-789",
        }

        FlextContext.set_from_context(headers)

        assert FlextContext.get_correlation_id() == "header-corr-123"
        assert FlextContext.get_parent_correlation_id() == "header-parent-456"
        assert FlextContext.get_service_name() == "header-service"
        assert FlextContext.get_user_id() == "header-user-789"

    def test_set_from_context_plain_format(self) -> None:
        """Test setting context from plain dictionary format."""
        plain_context = {
            "correlation_id": "plain-corr-123",
            "parent_correlation_id": "plain-parent-456",
            "service_name": "plain-service",
            "user_id": "plain-user-789",
        }

        FlextContext.set_from_context(plain_context)

        assert FlextContext.get_correlation_id() == "plain-corr-123"
        assert FlextContext.get_parent_correlation_id() == "plain-parent-456"
        assert FlextContext.get_service_name() == "plain-service"
        assert FlextContext.get_user_id() == "plain-user-789"

    def test_set_from_context_invalid_types(self) -> None:
        """Test setting context ignores invalid types."""
        invalid_context = {
            "X-Correlation-Id": 123,  # Not a string
            "X-Service-Name": None,  # None value
            "valid_key": "valid_value",
        }

        FlextContext.clear_context()
        FlextContext.set_from_context(invalid_context)

        # Only valid string values should be set
        assert FlextContext.get_correlation_id() is None
        assert FlextContext.get_service_name() is None


class TestContextUtilities:
    """Test context utility methods."""

    def test_ensure_correlation_id_exists(self) -> None:
        """Test ensure_correlation_id when ID exists."""
        existing_id = "existing-123"
        FlextContext.set_correlation_id(existing_id)

        result_id = FlextContext.ensure_correlation_id()
        assert result_id == existing_id

    def test_ensure_correlation_id_creates_new(self) -> None:
        """Test ensure_correlation_id creates new when none exists."""
        FlextContext.clear_context()

        result_id = FlextContext.ensure_correlation_id()
        assert result_id is not None
        assert len(result_id) == 36  # UUID4 format
        assert FlextContext.get_correlation_id() == result_id

    def test_has_correlation_id_true(self) -> None:
        """Test has_correlation_id returns True when ID exists."""
        FlextContext.set_correlation_id("test-123")
        assert FlextContext.has_correlation_id() is True

    def test_has_correlation_id_false(self) -> None:
        """Test has_correlation_id returns False when no ID."""
        FlextContext.clear_context()
        assert FlextContext.has_correlation_id() is False

    def test_get_context_summary_empty(self) -> None:
        """Test context summary when empty."""
        FlextContext.clear_context()
        summary = FlextContext.get_context_summary()
        assert summary == "FlextContext(empty)"

    def test_get_context_summary_populated(self) -> None:
        """Test context summary with data."""
        FlextContext.set_correlation_id("long-correlation-id-123456789")
        FlextContext.set_service_name("test-service")
        FlextContext.set_operation_name("test-operation")
        FlextContext.set_user_id("test-user")

        summary = FlextContext.get_context_summary()

        # Should contain truncated correlation ID and other values
        assert "correlation=long-cor..." in summary
        assert "service=test-service" in summary
        assert "operation=test-operation" in summary
        assert "user=test-user" in summary
        assert "FlextContext(" in summary

    def test_get_context_summary_partial(self) -> None:
        """Test context summary with partial data."""
        FlextContext.clear_context()
        FlextContext.set_service_name("only-service")

        summary = FlextContext.get_context_summary()
        assert "service=only-service" in summary
        assert summary == "FlextContext(service=only-service)"


class TestContextClear:
    """Test context clearing functionality."""

    def test_clear_context_all_variables(self) -> None:
        """Test clearing all context variables."""
        # Set all types of context variables
        FlextContext.set_correlation_id("test-corr")
        FlextContext.set_parent_correlation_id("test-parent")
        FlextContext.set_service_name("test-service")
        FlextContext.set_service_version("1.0.0")
        FlextContext.set_user_id("test-user")
        FlextContext.set_operation_name("test-operation")
        FlextContext.set_request_id("test-request")
        FlextContext.set_operation_start_time(datetime.now(UTC))
        FlextContext.set_operation_metadata({"key": "value"})

        # Clear all
        FlextContext.clear_context()

        # All should be None
        assert FlextContext.get_correlation_id() is None
        assert FlextContext.get_parent_correlation_id() is None
        assert FlextContext.get_service_name() is None
        assert FlextContext.get_service_version() is None
        assert FlextContext.get_user_id() is None
        assert FlextContext.get_operation_name() is None
        assert FlextContext.get_request_id() is None
        assert FlextContext.get_operation_start_time() is None
        assert FlextContext.get_operation_metadata() is None


class TestContextThreadSafety:
    """Test thread safety of context management."""

    def test_context_isolation_between_threads(self) -> None:
        """Test that context is isolated between threads."""
        results: dict[str, object] = {}

        def thread_worker(thread_id: str) -> None:
            # Each thread sets its own correlation ID
            FlextContext.set_correlation_id(f"thread-{thread_id}")
            FlextContext.set_service_name(f"service-{thread_id}")

            sleep(0.01)  # Small delay to allow interleaving

            # Each thread should see only its own values
            results[f"correlation_{thread_id}"] = FlextContext.get_correlation_id()
            results[f"service_{thread_id}"] = FlextContext.get_service_name()

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = Thread(target=thread_worker, args=(str(i),))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify each thread had isolated context
        assert results["correlation_0"] == "thread-0"
        assert results["correlation_1"] == "thread-1"
        assert results["correlation_2"] == "thread-2"
        assert results["service_0"] == "service-0"
        assert results["service_1"] == "service-1"
        assert results["service_2"] == "service-2"

    def test_context_inheritance_in_new_thread(self) -> None:
        """Test that context doesn't leak to new threads."""
        FlextContext.set_correlation_id("main-thread-id")

        new_thread_correlation = None

        def new_thread_worker() -> None:
            nonlocal new_thread_correlation
            new_thread_correlation = FlextContext.get_correlation_id()

        thread = Thread(target=new_thread_worker)
        thread.start()
        thread.join()

        # New thread should not inherit main thread's context
        assert new_thread_correlation is None
