"""Scheduler service for FLEXT Core."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class ScheduleType(Enum):
    """Schedule types."""

    ONCE = "once"
    RECURRING = "recurring"
    CRON = "cron"


class ScheduleStatus(Enum):
    """Schedule status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    COMPLETED = "completed"


class SchedulerService(ABC):
    """Abstract scheduler service."""

    @abstractmethod
    async def schedule_pipeline(
        self,
        pipeline_id: str,
        schedule_type: ScheduleType,
        schedule_config: dict[str, Any],
    ) -> str:
        """Schedule a pipeline execution."""

    @abstractmethod
    async def unschedule_pipeline(self, schedule_id: str) -> bool:
        """Remove a pipeline schedule."""

    @abstractmethod
    async def get_schedule(self, schedule_id: str) -> Optional[dict[str, Any]]:
        """Get schedule by ID."""

    @abstractmethod
    async def list_schedules(
        self, pipeline_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """List schedules."""

    @abstractmethod
    async def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a schedule."""

    @abstractmethod
    async def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule."""


class DefaultSchedulerService(SchedulerService):
    """Default implementation of scheduler service."""

    def __init__(self) -> None:
        self._schedules: dict[str, dict[str, Any]] = {}

    async def schedule_pipeline(
        self,
        pipeline_id: str,
        schedule_type: ScheduleType,
        schedule_config: dict[str, Any],
    ) -> str:
        """Schedule a pipeline execution."""
        schedule_id = str(uuid4())
        schedule = {
            "id": schedule_id,
            "pipeline_id": pipeline_id,
            "schedule_type": schedule_type.value,
            "config": schedule_config,
            "status": ScheduleStatus.ACTIVE.value,
            "created_at": datetime.now(),
            "next_run": self._calculate_next_run(schedule_type, schedule_config),
        }
        self._schedules[schedule_id] = schedule
        return schedule_id

    async def unschedule_pipeline(self, schedule_id: str) -> bool:
        """Remove a pipeline schedule."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            return True
        return False

    async def get_schedule(self, schedule_id: str) -> Optional[dict[str, Any]]:
        """Get schedule by ID."""
        return self._schedules.get(schedule_id)

    async def list_schedules(
        self, pipeline_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """List schedules."""
        schedules = list(self._schedules.values())
        if pipeline_id:
            schedules = [s for s in schedules if s["pipeline_id"] == pipeline_id]
        return schedules

    async def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a schedule."""
        schedule = self._schedules.get(schedule_id)
        if schedule:
            schedule["status"] = ScheduleStatus.PAUSED.value
            return True
        return False

    async def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule."""
        schedule = self._schedules.get(schedule_id)
        if schedule:
            schedule["status"] = ScheduleStatus.ACTIVE.value
            return True
        return False

    def _calculate_next_run(
        self, schedule_type: ScheduleType, config: dict[str, Any]
    ) -> Optional[datetime]:
        """Calculate next run time based on schedule configuration."""
        if schedule_type == ScheduleType.ONCE:
            return config.get("run_at")
        if schedule_type == ScheduleType.RECURRING:
            interval = config.get("interval_minutes", 60)
            return datetime.now() + timedelta(minutes=interval)
        if schedule_type == ScheduleType.CRON:
            # Would implement cron parsing here
            return datetime.now() + timedelta(hours=1)  # Placeholder
        return None
