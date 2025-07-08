"""Pytest configuration and shared fixtures.

Enterprise test configuration for 100% coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from flext_core import InMemoryRepository, Pipeline, PipelineService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for async tests."""
    return "asyncio"


@pytest_asyncio.fixture
async def repository() -> AsyncGenerator[InMemoryRepository[Pipeline]]:
    """Provide clean repository for each test."""
    repo = InMemoryRepository[Pipeline]()
    yield repo
    # Cleanup if needed


@pytest_asyncio.fixture
async def pipeline_service(
    repository: InMemoryRepository[Pipeline],
) -> AsyncGenerator[PipelineService]:
    """Provide pipeline service with repository."""
    service = PipelineService(repository)
    yield service
    # Cleanup if needed


@pytest.fixture
def mock_pipeline_data() -> dict[str, str]:
    """Provide mock pipeline data."""
    return {
        "name": "test-pipeline",
        "description": "Test pipeline for unit tests",
    }
