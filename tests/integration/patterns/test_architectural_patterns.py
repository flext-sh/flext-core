"""Advanced architectural pattern tests for FLEXT Core.

This module contains tests for architectural patterns including
Clean Architecture, DDD, CQRS, and enterprise design patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
from collections.abc import (
    Mapping,
    MutableMapping,
    MutableSequence,
)
from typing import Self

import pytest

from tests import c, m, p, r, t, u


class TestsFlextCoreArchitecturalPatterns:
    """Test enterprise design patterns."""

    @pytest.mark.architecture
    def test_factory_pattern_implementation(self) -> None:
        """Test Factory pattern implementation."""

        class ServiceFactory:
            """Factory for creating different types of services."""

            @staticmethod
            def create_service(service_type: str) -> p.Result[t.StrMapping]:
                """Create service based on type."""
                if service_type == "email":
                    return r[t.StrMapping].ok({
                        "type": "email",
                        "provider": "smtp",
                    })
                if service_type == "sms":
                    return r[t.StrMapping].ok({
                        "type": "sms",
                        "provider": "twilio",
                    })
                return r[t.StrMapping].fail(
                    f"Unknown service type: {service_type}",
                )

        email_service = ServiceFactory.create_service("email")
        assert email_service.success
        assert isinstance(email_service.value, dict)
        assert email_service.value["type"] == "email"
        sms_service = ServiceFactory.create_service("sms")
        assert sms_service.success
        assert isinstance(sms_service.value, dict)
        assert sms_service.value["type"] == "sms"
        invalid_service = ServiceFactory.create_service("invalid")
        assert invalid_service.failure

    @pytest.mark.architecture
    def test_builder_pattern_implementation(self) -> None:
        """Test Builder pattern implementation."""

        class ConfigurationBuilder:
            """Builder for complex configuration objects."""

            def __init__(self) -> None:
                """Initialize builder."""
                super().__init__()
                self.config: dict[str, t.JsonValue] = {}

            def with_database(self, host: str, port: int) -> Self:
                """Add database configuration."""
                self.config["database"] = {"host": host, "port": port}
                return self

            def with_logging(self, level: str) -> Self:
                """Add logging configuration."""
                self.config["logging"] = {"level": level}
                return self

            def with_cache(self, *, enabled: bool) -> Self:
                """Add cache configuration."""
                self.config["cache"] = {"enabled": enabled}
                return self

            def build(self) -> p.Result[Mapping[str, t.JsonValue]]:
                """Build the configuration."""
                if not self.config:
                    return r[Mapping[str, t.JsonValue]].fail(
                        "Configuration cannot be empty",
                    )
                return r[Mapping[str, t.JsonValue]].ok(dict(self.config))

        config_result = (
            ConfigurationBuilder()
            .with_database(c.LOCALHOST, 5432)
            .with_logging("INFO")
            .with_cache(enabled=True)
            .build()
        )
        assert config_result.success
        settings = config_result.value
        assert isinstance(settings, dict)
        database = settings.get("database")
        assert isinstance(database, dict)
        assert database.get("host") == c.LOCALHOST
        logging = settings.get("logging")
        assert isinstance(logging, dict)
        assert logging.get("level") == "INFO"
        cache = settings.get("cache")
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
                self._data: MutableMapping[str, m.ConfigMap] = {}
                self._query_count = 0

            def save(self, entity_id: str, data: m.ConfigMap) -> p.Result[bool]:
                """Save entity to repository."""
                self._data[entity_id] = data
                return r[bool].ok(True)

            def find_by_id(self, entity_id: str) -> p.Result[m.ConfigMap]:
                """Find entity by ID."""
                self._query_count += 1
                if entity_id in self._data:
                    return r[m.ConfigMap].ok(self._data[entity_id])
                return r[m.ConfigMap].fail(
                    f"Entity not found: {entity_id}",
                )

            def get_query_count(self) -> int:
                """Get number of queries executed."""
                return self._query_count

        repo = InMemoryRepository()
        start_time = time.perf_counter()
        for i in range(1000):
            result = repo.save(
                f"entity_{i}",
                m.ConfigMap(root={"id": i, "name": f"Entity {i}"}),
            )
            u.Tests.assert_success(
                result,
                f"Save operation {i} should succeed",
            )
        save_duration = time.perf_counter() - start_time
        assert save_duration < 1.0, (
            f"1000 saves took {save_duration:.3f}s, expected < 1.0s"
        )
        assert save_duration > 0, "Save duration should be positive"
        assert len(repo._data) == 1000, f"Expected 1000 entities, got {len(repo._data)}"
        start_time = time.perf_counter()
        for i in range(100):
            query_result: p.Result[m.ConfigMap] = repo.find_by_id(f"entity_{i}")
            assert query_result.success, f"Query {i} should succeed"
            entity_data = query_result.value
            assert isinstance(entity_data, m.ConfigMap), (
                f"Expected ConfigMap, got {type(entity_data)}"
            )
            assert entity_data.root.get("id") == i, f"Entity {i} should have id={i}"
        query_duration = time.perf_counter() - start_time
        assert query_duration < 0.5, (
            f"100 queries took {query_duration:.3f}s, expected < 0.5s"
        )
        assert repo.get_query_count() == 100, (
            f"Expected 100 queries, got {repo.get_query_count()}"
        )
        query_duration = time.perf_counter() - start_time
        assert query_duration < 0.5, (
            f"100 queries took {query_duration:.3f}s, expected < 0.5s"
        )
        assert query_duration > 0, "Query duration should be positive"
        assert repo.get_query_count() == 100, (
            f"Expected 100 queries, got {repo.get_query_count()}"
        )

    @pytest.mark.architecture
    @pytest.mark.ddd
    def test_domain_event_pattern(self) -> None:
        """Test Domain Event pattern implementation."""

        class UserEventHandler:
            """Handler for user domain events."""

            def __init__(self) -> None:
                """Initialize handler."""
                super().__init__()
                self.processed_events: MutableSequence[m.DomainEvent] = []

            def handle_user_created(
                self, event: m.Tests.UserCreatedEvent
            ) -> p.Result[bool]:
                """Handle user created event."""
                self.processed_events.append(event)
                return r[bool].ok(True)

            def handle_user_updated(
                self, event: m.Tests.UserUpdatedEvent
            ) -> p.Result[bool]:
                """Handle user updated event."""
                self.processed_events.append(event)
                return r[bool].ok(True)

        handler = UserEventHandler()
        created_event = m.Tests.UserCreatedEvent.model_validate({
            "event_type": "UserCreated",
            "aggregate_id": "user_123",
            "user_id": "123",
            "user_name": "John Doe",
            "timestamp": time.time(),
        })
        updated_event = m.Tests.UserUpdatedEvent.model_validate({
            "event_type": "UserUpdated",
            "aggregate_id": "user_123",
            "user_id": "123",
            "old_name": "John Doe",
            "new_name": "Jane Doe",
            "timestamp": time.time(),
        })
        result1 = handler.handle_user_created(created_event)
        assert result1.success
        result2 = handler.handle_user_updated(updated_event)
        assert result2.success
        assert len(handler.processed_events) == 2
        assert isinstance(handler.processed_events[0], m.Tests.UserCreatedEvent)
        assert isinstance(handler.processed_events[1], m.Tests.UserUpdatedEvent)

    @pytest.mark.architecture
    def test_observer_pattern_implementation(self) -> None:
        """Test Observer pattern implementation."""
        observers: MutableSequence[t.MutableJsonMapping] = []

        def notify_all(state: str) -> None:
            for observer in observers:
                observer["state"] = state

        obs1: t.MutableJsonMapping = {
            "name": "Observer1",
            "state": "",
        }
        obs2: t.MutableJsonMapping = {
            "name": "Observer2",
            "state": "",
        }
        observers.extend([obs1, obs2])
        notify_all("new_state")
        assert obs1["state"] == "new_state"
        assert obs2["state"] == "new_state"
        observers.remove(obs1)
        notify_all("updated_state")
        assert obs1["state"] == "new_state"
        assert obs2["state"] == "updated_state"
