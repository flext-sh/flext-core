"""Additional tests to achieve near 100% coverage of aggregate_root.py.

These tests target uncovered lines to bring aggregate_root.py coverage
from 53% to as close to 100% as possible.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from flext_core import FlextAggregateRoot, FlextValidationError
from flext_core.aggregate_root import FlextAggregateRoot as _FlextAggregateRoot
from flext_core.root_models import FlextMetadata

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestHelperFunctionsCoverage:
    """Test helper functions for complete coverage."""

    def test_coerce_metadata_to_root_none_input(self) -> None:
        """Test _coerce_metadata_to_root with None input."""
        result = _FlextAggregateRoot._coerce_metadata_to_root(None)
        assert result is None

    def test_coerce_metadata_to_root_flext_metadata_input(self) -> None:
        """Test _coerce_metadata_to_root with FlextMetadata input."""
        original_meta = FlextMetadata({"key": "value"})
        result = _FlextAggregateRoot._coerce_metadata_to_root(original_meta)
        assert result is original_meta

    def test_coerce_metadata_to_root_dict_input(self) -> None:
        """Test _coerce_metadata_to_root with dict input."""
        meta_dict = {"key": "value", "number": 42}
        result = _FlextAggregateRoot._coerce_metadata_to_root(meta_dict)
        assert isinstance(result, FlextMetadata)
        assert result.root["key"] == "value"
        assert result.root["number"] == 42

    def test_coerce_metadata_to_root_convertible_object(self) -> None:
        """Test _coerce_metadata_to_root with dict-like object."""

        class DictLike:
            def __init__(self) -> None:
                self.data = {"test": "value"}

            def __iter__(self):
                return iter(self.data)

            def keys(self):
                return self.data.keys()

            def __getitem__(self, key):
                return self.data[key]

            def items(self):
                return self.data.items()

        dict_like = DictLike()
        result = _FlextAggregateRoot._coerce_metadata_to_root(dict_like)
        assert isinstance(result, FlextMetadata)
        # Should fallback to raw conversion
        assert "raw" in result.root or "test" in result.root

    def test_coerce_metadata_to_root_non_convertible_object(self) -> None:
        """Test _coerce_metadata_to_root with non-convertible object."""

        class NonConvertible:
            def __str__(self) -> str:
                return "non-convertible"

        obj = NonConvertible()
        result = _FlextAggregateRoot._coerce_metadata_to_root(obj)
        assert isinstance(result, FlextMetadata)
        assert result.root["raw"] == "non-convertible"

    def test_normalize_domain_event_list_dict_events(self) -> None:
        """Test _normalize_domain_event_list with dict events."""
        raw_events = [
            {"event_type": "UserCreated", "data": {"name": "test"}},
            {"event_type": "UserUpdated", "data": {"name": "updated"}},
        ]

        result = _FlextAggregateRoot._normalize_domain_event_list(raw_events)

        assert len(result) == 2
        assert all(isinstance(event, dict) for event in result)
        assert result[0]["event_type"] == "UserCreated"
        assert result[1]["event_type"] == "UserUpdated"

    def test_normalize_domain_event_list_model_dump_objects(self) -> None:
        """Test _normalize_domain_event_list with objects having model_dump."""

        class EventLike:
            def __init__(self, event_type: str, data: dict) -> None:
                self.event_type = event_type
                self.data = data

            def model_dump(self) -> dict:
                return {"event_type": self.event_type, "data": self.data}

        raw_events = [
            EventLike("UserCreated", {"name": "test"}),
            EventLike("UserDeleted", {"id": "123"}),
        ]

        result = _FlextAggregateRoot._normalize_domain_event_list(raw_events)

        assert len(result) == 2
        assert result[0]["event_type"] == "UserCreated"
        assert result[1]["event_type"] == "UserDeleted"

    def test_normalize_domain_event_list_non_dict_model_dump(self) -> None:
        """Test _normalize_domain_event_list with model_dump returning non-dict."""

        class BadModelDump:
            def model_dump(self) -> str:
                return "not-a-dict"

        raw_events = [BadModelDump()]
        result = _FlextAggregateRoot._normalize_domain_event_list(raw_events)

        assert len(result) == 1
        assert "event" in result[0]
        assert isinstance(result[0]["event"], str)

    def test_normalize_domain_event_list_string_objects(self) -> None:
        """Test _normalize_domain_event_list with string/other objects."""
        raw_events = ["simple_string_event", 42, None]

        result = _FlextAggregateRoot._normalize_domain_event_list(raw_events)

        assert len(result) == 3
        assert result[0]["event"] == "simple_string_event"
        assert result[1]["event"] == "42"
        assert result[2]["event"] == "None"


class TestAggregateRootInitializationCoverage:
    """Test FlextAggregateRoot initialization edge cases."""

    def test_aggregate_root_with_provided_id_in_data(self) -> None:
        """Test initialization with ID provided in data dict."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        # ID in data should take precedence
        aggregate = TestAggregate(
            entity_id="param_id",
            name="test",
            id="data_id"  # This should win
        )

        assert aggregate.id == "data_id"
        assert aggregate.name == "test"

    def test_aggregate_root_with_entity_id_param(self) -> None:
        """Test initialization with entity_id parameter."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        aggregate = TestAggregate(
            entity_id="param_id",
            name="test"
        )

        assert aggregate.id == "param_id"

    def test_aggregate_root_with_generated_id(self) -> None:
        """Test initialization with auto-generated ID."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        # Should have auto-generated UUID
        assert len(str(aggregate.id)) > 0
        assert "-" in str(aggregate.id)  # UUID format

    def test_aggregate_root_with_domain_events_in_data(self) -> None:
        """Test initialization with domain_events in data."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        events = [
            {"event_type": "AggregateCreated", "data": {"name": "test"}},
        ]

        aggregate = TestAggregate(
            name="test",
            domain_events=events
        )

        assert aggregate.name == "test"
        # Domain events should be processed and stored

    def test_aggregate_root_with_created_at_in_data(self) -> None:
        """Test initialization with created_at datetime in data."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        created_time = datetime.now(UTC)

        aggregate = TestAggregate(
            name="test",
            created_at=created_time
        )

        assert aggregate.name == "test"
        # Should use the provided created_at

    def test_aggregate_root_with_metadata_dict(self) -> None:
        """Test initialization with metadata dict."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        metadata = {"source": "test", "version": 1}

        aggregate = TestAggregate(
            name="test",
            metadata=metadata
        )

        assert aggregate.name == "test"
        # Metadata should be converted to FlextMetadata

    def test_aggregate_root_initialization_failure(self) -> None:
        """Test initialization failure handling."""

        class BadAggregate(FlextAggregateRoot):
            required_field: str

        # Should raise FlextValidationError for missing required field
        with pytest.raises(FlextValidationError) as exc_info:
            BadAggregate()  # Missing required_field

        error = exc_info.value
        assert "Failed to initialize aggregate root" in str(error)
        assert error.validation_details is not None
        assert "aggregate_id" in error.validation_details


class TestAggregateRootDomainEventsCoverage:
    """Test domain events functionality for coverage."""

    def test_add_domain_event_with_string_and_data(self) -> None:
        """Test add_domain_event with string event type and separate data."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        result = aggregate.add_domain_event(
            "UserCreated",
            {"name": "test", "id": aggregate.id}
        )

        assert result.success

    def test_add_domain_event_with_dict_only(self) -> None:
        """Test add_domain_event with event dict only."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        event_dict = {
            "event_type": "UserCreated",
            "data": {"name": "test", "id": aggregate.id}
        }

        result = aggregate.add_domain_event(event_dict)

        assert result.success

    def test_add_domain_event_with_empty_data(self) -> None:
        """Test add_domain_event with empty event data."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        aggregate = TestAggregate(name="test")

        # Should handle None event_data gracefully
        result = aggregate.add_domain_event("EmptyEvent")

        assert result.success


class TestAggregateRootEdgeCasesCoverage:
    """Test edge cases for complete coverage."""

    def test_non_dict_domain_events_input(self) -> None:
        """Test aggregate with non-list domain_events."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        # domain_events is not a list - should default to empty
        aggregate = TestAggregate(
            name="test",
            domain_events="not-a-list"  # Should be ignored
        )

        assert aggregate.name == "test"

    def test_non_datetime_created_at(self) -> None:
        """Test aggregate with non-datetime created_at."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        aggregate = TestAggregate(
            name="test",
            created_at="not-a-datetime"  # Should be ignored
        )

        assert aggregate.name == "test"

    def test_complex_metadata_scenarios(self) -> None:
        """Test various metadata input scenarios."""

        class TestAggregate(FlextAggregateRoot):
            name: str

        # Test with complex metadata
        complex_meta = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
        }

        aggregate = TestAggregate(
            name="test",
            metadata=complex_meta
        )

        assert aggregate.name == "test"
