"""Entity coverage tests targeting lines 82, 87, 110, 139, 235.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c, m, r


class TestEntityCoverageEdgeCases:
    """Tests for entity.py property returns and edge case handling."""

    def test_entity_id_property(self) -> None:
        """entity_id returns unique_id (line 82)."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(unique_id="test-id-123", name="test")
        assert entity.entity_id == "test-id-123"
        assert entity.entity_id == entity.unique_id

    def test_logger_property(self) -> None:
        """logger property returns FlextRuntime.get_logger (line 87)."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(unique_id="test-id", name="test")
        logger = entity.logger
        assert logger is not None
        assert hasattr(logger, "info") or hasattr(logger, "debug")

    def test_uncommitted_events_property(self) -> None:
        """uncommitted_events returns list(domain_events) (line 110)."""

        class TestEntity(m.Entity):
            name: str

        entity = TestEntity(unique_id="test-id", name="test")
        events = entity.uncommitted_events
        assert isinstance(events, list)
        assert len(events) == 0

        event_result = entity.add_domain_event("test_event", {"key": "value"})
        assert event_result.is_success

        events = entity.uncommitted_events
        assert len(events) == 1
        assert events[0].event_type == "test_event"

    def test_add_domain_event_max_events_error(self) -> None:
        """add_domain_event fails exceeding max limit (line 139)."""

        class TestEntry(m.Entity):
            name: str

        entry = TestEntry(unique_id="test-id", name="test")
        max_events = c.Validation.MAX_UNCOMMITTED_EVENTS
        for i in range(max_events):
            result = entry.add_domain_event(f"event_{i}", {})
            assert result.is_success

        fail_result = entry.add_domain_event("overflow_event", {})
        assert fail_result.is_failure
        error_msg = fail_result.error or ""
        assert "would exceed max events limit" in error_msg

    def test_value_eq_not_implemented(self) -> None:
        """Value.__eq__ returns NotImplemented for non-BaseModel (line 235)."""

        class TestValue(m.Value):
            data: str

        value = TestValue(data="test")
        assert value.__eq__("not a model") is NotImplemented
        assert value.__eq__(123) is NotImplemented
        assert value.__eq__(None) is NotImplemented

        value2 = TestValue(data="test")
        assert value == value2

        value3 = TestValue(data="different")
        assert value != value3
