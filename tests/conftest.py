"""Enterprise-grade test configuration for FLX-Core.

Adapted from flx-meltano-enterprise with proper imports and fixtures.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Ensure the src directory is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flx_core.domain.entities import (
    Pipeline,
    PipelineExecution,
    PipelineId,
    PipelineName,
    PipelineStep,
    Plugin,
    PluginConfiguration,
    PluginId,
    PluginType,
)
from flx_core.domain.value_objects import ExecutionStatus
from flx_core.infrastructure.persistence.models import Base
from flx_core.infrastructure.persistence.unit_of_work import UnitOfWork

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

# Python 3.13 type aliases
type TestDatabase = str
type TestPipeline = Pipeline
type TestExecution = PipelineExecution
type TestPlugin = Plugin


# === ENVIRONMENT SETUP ===


def _setup_test_environment() -> None:
    """Set up test environment with proper isolation."""
    test_env_path = Path(__file__).parent / ".env"

    # Load test environment if exists
    if test_env_path.exists():
        load_dotenv(test_env_path, override=True)

    # Set testing-specific overrides
    os.environ["FLX_ENVIRONMENT"] = "testing"
    os.environ["FLX_TEST_MODE"] = "true"
    os.environ["FLX_DEBUG"] = "false"

    # Mock external services
    os.environ["FLX_TEST_MOCK_EXTERNAL_SERVICES"] = "true"
    os.environ["FLX_TEST_MOCK_REDIS"] = "true"


# Setup environment on import
_setup_test_environment()


# === FIXTURES ===


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_database() -> AsyncIterator[TestDatabase]:
    """Create test database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_db_file:
        temp_db_path = temp_db_file.name

    # Override database URL
    original_db_url = os.environ.get("FLX_DATABASE_URL")
    os.environ["FLX_DATABASE_URL"] = f"sqlite+aiosqlite:///{temp_db_path}"

    # Create engine and tables
    engine = create_async_engine(
        os.environ["FLX_DATABASE_URL"],
        echo=False,
        future=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield temp_db_path

    # Cleanup
    await engine.dispose()
    if original_db_url:
        os.environ["FLX_DATABASE_URL"] = original_db_url
    elif "FLX_DATABASE_URL" in os.environ:
        del os.environ["FLX_DATABASE_URL"]

    if Path(temp_db_path).exists():
        os.unlink(temp_db_path)


@pytest.fixture
async def db_session(test_database: TestDatabase) -> AsyncIterator[AsyncSession]:
    """Create database session for tests."""
    engine = create_async_engine(
        os.environ["FLX_DATABASE_URL"],
        echo=False,
        future=True,
    )

    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def unit_of_work(db_session: AsyncSession) -> UnitOfWork:
    """Create unit of work for tests."""
    return UnitOfWork(session=db_session)


# === DOMAIN FIXTURES ===


@pytest.fixture
def sample_pipeline_id() -> PipelineId:
    """Create sample pipeline ID."""
    return PipelineId()


@pytest.fixture
def sample_pipeline_name() -> PipelineName:
    """Create sample pipeline name."""
    return PipelineName(value="test_pipeline")


@pytest.fixture
def sample_plugin_id() -> PluginId:
    """Create sample plugin ID."""
    return PluginId()


@pytest.fixture
def sample_plugin() -> Plugin:
    """Create sample plugin."""
    return Plugin(
        plugin_id=PluginId(),
        plugin_type=PluginType.EXTRACTOR,
        name="test_plugin",
        description="Test plugin for testing",
        configuration=PluginConfiguration(
            plugin_name="test_plugin",
            namespace="test",
            pip_url="test-plugin==1.0.0",
            executable="test-plugin",
            settings={"test": "value"},
        ),
    )


@pytest.fixture
def sample_pipeline(sample_pipeline_id: PipelineId, sample_plugin: Plugin) -> Pipeline:
    """Create sample pipeline with steps."""
    pipeline = Pipeline(
        pipeline_id=sample_pipeline_id,
        name=PipelineName(value="test_pipeline"),
        description="Test pipeline for testing",
    )

    # Add a step
    step = PipelineStep(
        name="extract",
        plugin=sample_plugin,
        configuration={"source": "test"},
        depends_on=[],
    )
    pipeline.add_step(step)

    return pipeline


@pytest.fixture
def sample_execution(sample_pipeline: Pipeline) -> PipelineExecution:
    """Create sample pipeline execution."""
    return PipelineExecution(
        pipeline_id=sample_pipeline.pipeline_id,
        status=ExecutionStatus.PENDING,
    )


# === TEST UTILITIES ===


@pytest.fixture
def mock_async():
    """Helper to create async mocks."""
    from unittest.mock import AsyncMock

    return AsyncMock


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for async tests."""
    return "asyncio"
