"""Main test module - Quick smoke tests.

For comprehensive tests, see unit/ and integration/ directories.
"""

from __future__ import annotations

import pytest

from flext_core import (
    PipelineName,
    ServiceResult,
    __version__,
)


def test_version() -> None:
    """Test package version is accessible."""
    assert __version__ == "0.6.0"


def test_basic_imports() -> None:
    """Test all main imports work correctly."""
    # Domain imports
    from flext_core import (
        AggregateRoot,
        InMemoryRepository,
        PipelineService,
    )

    # All imports should work
    assert AggregateRoot is not None
    assert PipelineService is not None
    assert InMemoryRepository is not None


def test_quick_pipeline_name_validation() -> None:
    """Quick test for pipeline name validation."""
    # Valid name
    name = PipelineName(value="test-pipeline")
    assert str(name) == "test-pipeline"

    # Invalid name
    with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
        PipelineName(value="   ")


def test_quick_service_result() -> None:
    """Quick test for ServiceResult pattern."""
    # Success case
    success: ServiceResult[int] = ServiceResult.ok(42)
    assert success.success
    assert success.unwrap() == 42

    # Failure case
    failure: ServiceResult[int] = ServiceResult.fail("error")
    assert not failure.success
    assert failure.error == "error"

    with pytest.raises(RuntimeError, match="error"):
        failure.unwrap()
