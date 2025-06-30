"""Meltano wrapper utilities."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RefreshMode(Enum):
    """Refresh mode for Meltano operations."""

    FULL = "full"
    INCREMENTAL = "incremental"
    AUTO = "auto"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    name: str
    plugins: list[str]


@dataclass
class PluginFilter:
    """Filter for plugin selection."""

    include: list[str] | None = None
    exclude: list[str] | None = None


@dataclass
class MeltanoExecutionResult:
    """Result of Meltano execution."""

    success: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0


class MeltanoEngine:
    """Mock Meltano engine for compatibility."""

    def __init__(self) -> None:
        pass

    async def run_pipeline(self, config: PipelineConfig) -> dict[str, Any]:
        """Run a Meltano pipeline."""
        return {"status": "success", "pipeline": config.name}

    async def execute(self, command: list[str]) -> MeltanoExecutionResult:
        """Execute a Meltano command."""
        return MeltanoExecutionResult(success=True, exit_code=0)
