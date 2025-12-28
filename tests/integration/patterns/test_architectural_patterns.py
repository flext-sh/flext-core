"""Advanced architectural pattern tests for FLEXT Core.

This module contains tests for architectural patterns including
Clean Architecture, DDD, CQRS, and enterprise design patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time

import pytest

from flext_core import FlextResult, m
from flext_core.typings import t
from tests.test_utils import assertion_helpers


class TestEnterprisePatterns:
    """Test enterprise design patterns."""

    @pytest.mark.architecture
    def test_factory_pattern_implementation(self) -> None:
        """Test Factory pattern implementation."""

        class ServiceFactory:
            """Factory for creating different types of services."""

            @staticmethod
            def create_service(
                service_type: str,
            ) -> FlextResult[dict[str, str]]:
                """Create service based on type."""
                if service_type == "email":
                    return FlextResult[dict[str, str]].ok(
                        {
                            "type": "email",
                            "provider": "smtp",
                        },
                    )
                if service_type == "sms":
                    return FlextResult[dict[str, str]].ok(
                        {
                            "type": "sms",
                            "provider": "twilio",
                        },
                    )
                return FlextResult[dict[str, str]].fail(
                    f"Unknown service type: {service_type}",
                )

        # Test factory usage
        email_service = ServiceFactory.create_service("email")
        assert email_service.is_success
        assert isinstance(email_service.value, dict)
        assert email_service.value["type"] == "email"

        sms_service = ServiceFactory.create_service("sms")
        assert sms_service.is_success
        assert isinstance(sms_service.value, dict)
        assert sms_service.value["type"] == "sms"

        invalid_service = ServiceFactory.create_service("invalid")
        assert invalid_service.is_failure

    @pytest.mark.architecture
    def test_builder_pattern_implementation(self) -> None:
        """Test Builder pattern implementation."""

        class ConfigurationBuilder:
            """Builder for complex configuration objects."""

            def __init__(self) -> None:
                """Initialize builder."""
                super().__init__()
                self._config: dict[str, t.GeneralValueType] = {}

            def with_database(self, host: str, port: int) -> ConfigurationBuilder:
                """Add database configuration."""
                self._config["database"] = {"host": host, "port": port}
                return self

            def with_logging(self, level: str) -> ConfigurationBuilder:
                """Add logging configuration."""
                self._config["logging"] = {"level": level}
                return self

            def with_cache(self, *, enabled: bool) -> ConfigurationBuilder:
                """Add cache configuration."""
                self._config["cache"] = {"enabled": enabled}
                return self

            def build(self) -> FlextResult[dict[str, t.GeneralValueType]]:
                """Build the configuration."""
                if not self._config:
                    return FlextResult[dict[str, t.GeneralValueType]].fail(
                        "Configuration cannot be empty",
                    )

                return FlextResult[dict[str, t.GeneralValueType]].ok(self._config.copy())

        # Test builder usage
        config_result = (
            ConfigurationBuilder()
            .with_database("localhost", 5432)
            .with_logging("INFO")
            .with_cache(enabled=True)
            .build()
        )

        assert config_result.is_success
        config = config_result.value
        assert isinstance(config, dict)
        # Verify database config
        database = config.get("database")
        assert isinstance(database, dict)
        assert database.get("host") == "localhost"
        # Verify logging config
        logging = config.get("logging")
        assert isinstance(logging, dict)
        assert logging.get("level") == "INFO"
        # Verify cache config
        cache = config.get("cache")
        assert isinstance(cache, dict)
        assert cache.get("enabled")

    @pytest.mark.architecture
    @pytest.mark.performance
    def test_repository_pattern_performance(self) -> None:
        """Test Repository pattern with performance considerations."""

        class InMemoryRepository:
            """In-memory repository implementation."""

            def __init__(self) -> None:
                """Initialize repository."""
                super().__init__()
                self._data: dict[str, t.GeneralValueType] = {}
                self._query_count = 0

            def save(self, entity_id: str, data: object) -> FlextResult[bool]:
                """Save entity to repository."""
                self._data[entity_id] = data
                return FlextResult[bool].ok(True)

            def find_by_id(self, entity_id: str) -> FlextResult[object]:
                """Find entity by ID."""
                self._query_count += 1

                if entity_id in self._data:
                    return FlextResult[object].ok(self._data[entity_id])

                return FlextResult[object].fail(f"Entity not found: {entity_id}")

            def get_query_count(self) -> int:
                """Get number of queries executed."""
                return self._query_count

        # Test repository performance
        repo = InMemoryRepository()

        # Save multiple entities
        start_time = time.perf_counter()
        for i in range(1000):
            result = repo.save(f"entity_{i}", {"id": i, "name": f"Entity {i}"})
            (
                assertion_helpers.assert_flext_result_success(result),
                f"Save operation {i} should succeed",
            )

        save_duration = time.perf_counter() - start_time

        # Validate performance: 1000 saves should complete in reasonable time
        assert save_duration < 1.0, (
            f"1000 saves took {save_duration:.3f}s, expected < 1.0s"
        )
        assert save_duration > 0, "Save duration should be positive"

        # Validate all entities were saved
        assert len(repo._data) == 1000, f"Expected 1000 entities, got {len(repo._data)}"

        # Query entities
        start_time = time.perf_counter()
        for i in range(100):
            query_result: FlextResult[object] = repo.find_by_id(f"entity_{i}")
            assert query_result.is_success, f"Query {i} should succeed"
            entity_data = query_result.value
            assert isinstance(entity_data, dict), f"Expected dict, got {type(entity_data)}"
            assert entity_data.get("id") == i, f"Entity {i} should have id={i}"

        query_duration = time.perf_counter() - start_time

        # Validate performance: 100 queries should complete in reasonable time
        assert query_duration < 0.5, (
            f"100 queries took {query_duration:.3f}s, expected < 0.5s"
        )

        # Validate query count
        assert repo.get_query_count() == 100, (
            f"Expected 100 queries, got {repo.get_query_count()}"
        )

        query_duration = time.perf_counter() - start_time

        # Validate performance: 100 queries should complete in reasonable time
        assert query_duration < 0.5, (
            f"100 queries took {query_duration:.3f}s, expected < 0.5s"
        )
        assert query_duration > 0, "Query duration should be positive"

        # Validate query count
        assert repo.get_query_count() == 100, (
            f"Expected 100 queries, got {repo.get_query_count()}"
        )

        # Validate all queries succeeded and returned correct data
        # (already validated in loop above)


class TestEventDrivenPatterns:
    """Test event-driven architectural patterns."""

    @pytest.mark.architecture
    @pytest.mark.ddd
    def test_domain_event_pattern(self) -> None:
        """Test Domain Event pattern implementation."""

        # Event classes
        class UserCreatedEvent(m.DomainEvent):
            """Domain event for user creation using FlextModels foundation."""

            user_id: str
            user_name: str
            timestamp: float

        class UserUpdatedEvent(m.DomainEvent):
            """Domain event for user updates."""

            user_id: str
            old_name: str
            new_name: str
            timestamp: float

        # Event handler
        class UserEventHandler:
            """Handler for user domain events."""

            def __init__(self) -> None:
                """Initialize handler."""
                super().__init__()
                self.processed_events: list[m.DomainEvent] = []

            def handle_user_created(self, event: UserCreatedEvent) -> FlextResult[bool]:
                """Handle user created event."""
                self.processed_events.append(event)
                return FlextResult[bool].ok(True)

            def handle_user_updated(self, event: UserUpdatedEvent) -> FlextResult[bool]:
                """Handle user updated event."""
                self.processed_events.append(event)
                return FlextResult[bool].ok(True)

        # Test event processing
        handler = UserEventHandler()

        # Create and process events with required fields
        created_event = UserCreatedEvent(
            event_type="UserCreated",
            aggregate_id="user_123",
            user_id="123",
            user_name="John Doe",
            timestamp=time.time(),
        )

        updated_event = UserUpdatedEvent(
            event_type="UserUpdated",
            aggregate_id="user_123",
            user_id="123",
            old_name="John Doe",
            new_name="Jane Doe",
            timestamp=time.time(),
        )

        # Process events
        result1 = handler.handle_user_created(created_event)
        assert result1.is_success

        result2 = handler.handle_user_updated(updated_event)
        assert result2.is_success

        # Verify event processing
        assert len(handler.processed_events) == 2
        assert isinstance(handler.processed_events[0], UserCreatedEvent)
        assert isinstance(handler.processed_events[1], UserUpdatedEvent)

    @pytest.mark.architecture
    def test_observer_pattern_implementation(self) -> None:
        """Test Observer pattern implementation."""
        observers: list[dict[str, t.GeneralValueType]] = []

        def notify_all(state: str) -> None:
            for observer in observers:
                observer["state"] = state

        # Create observers
        obs1: dict[str, t.GeneralValueType] = {"name": "Observer1", "state": None}
        obs2: dict[str, t.GeneralValueType] = {"name": "Observer2", "state": None}
        observers.extend([obs1, obs2])

        # Test notifications
        notify_all("new_state")
        assert obs1["state"] == "new_state"
        assert obs2["state"] == "new_state"

        # Test removal
        observers.remove(obs1)
        notify_all("updated_state")
        assert obs1["state"] == "new_state"  # Not updated
        assert obs2["state"] == "updated_state"  # Updated
