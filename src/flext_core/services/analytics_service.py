"""Analytics service for FLEXT Core."""

from abc import ABC, abstractmethod
from typing import Any


class AnalyticsService(ABC):
    """Abstract analytics service."""

    @abstractmethod
    async def track_event(self, event_name: str, properties: dict[str, Any]) -> None:
        """Track an analytics event."""

    @abstractmethod
    async def track_pipeline_execution(
        self, pipeline_id: str, execution_id: str, properties: dict[str, Any]
    ) -> None:
        """Track pipeline execution analytics."""

    @abstractmethod
    async def get_pipeline_metrics(self, pipeline_id: str) -> dict[str, Any]:
        """Get analytics metrics for a pipeline."""

    @abstractmethod
    async def get_system_metrics(self) -> dict[str, Any]:
        """Get system-wide analytics metrics."""


class DefaultAnalyticsService(AnalyticsService):
    """Default implementation of analytics service."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    async def track_event(self, event_name: str, properties: dict[str, Any]) -> None:
        """Track an analytics event."""
        event = {
            "event_name": event_name,
            "properties": properties,
            "timestamp": None,  # Would use actual timestamp in real implementation
        }
        self._events.append(event)

    async def track_pipeline_execution(
        self, pipeline_id: str, execution_id: str, properties: dict[str, Any]
    ) -> None:
        """Track pipeline execution analytics."""
        await self.track_event(
            "pipeline_execution",
            {
                "pipeline_id": pipeline_id,
                "execution_id": execution_id,
                **properties,
            },
        )

    async def get_pipeline_metrics(self, pipeline_id: str) -> dict[str, Any]:
        """Get analytics metrics for a pipeline."""
        pipeline_events = [
            e
            for e in self._events
            if e.get("properties", {}).get("pipeline_id") == pipeline_id
        ]
        return {
            "total_events": len(pipeline_events),
            "event_types": list({e["event_name"] for e in pipeline_events}),
        }

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get system-wide analytics metrics."""
        return {
            "total_events": len(self._events),
            "unique_event_types": len({e["event_name"] for e in self._events}),
        }
