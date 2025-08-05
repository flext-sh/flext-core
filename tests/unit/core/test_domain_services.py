"""Tests for FLEXT Core domain services module."""

from __future__ import annotations

import pytest

from flext_core.domain_services import FlextDomainService
from flext_core.result import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


class ConcreteDomainService(FlextDomainService[str]):
    """Concrete implementation for testing."""

    def execute(self) -> FlextResult[str]:
        """Execute the domain service."""
        return FlextResult.ok("executed")


class TestFlextDomainService:
    """Test FlextDomainService functionality."""

    def test_domain_service_creation(self) -> None:
        """Test FlextDomainService instantiation."""
        service = ConcreteDomainService()

        # Should be a valid instance
        assert isinstance(service, FlextDomainService)
        assert isinstance(service, ConcreteDomainService)

    def test_domain_service_execution(self) -> None:
        """Test domain service execution."""
        service = ConcreteDomainService()

        # Should be able to execute
        result = service.execute()
        assert result.success
        assert result.data == "executed", f"Expected {'executed'}, got {result.data}"

    def test_abstract_method_coverage(self) -> None:
        """Test coverage of abstract method definition."""
        # This covers the TYPE_CHECKING import for FlextResult

        # Verify abstract class structure
        assert hasattr(FlextDomainService, "execute")
        assert FlextDomainService.__abstractmethods__ == frozenset(["execute"]), (
            f"Expected {frozenset(['execute'])}, got {FlextDomainService.__abstractmethods__}"
        )

    def test_domain_service_inheritance_structure(self) -> None:
        """Test domain service class hierarchy."""
        service = ConcreteDomainService()

        # Should inherit from base classes properly
        assert issubclass(ConcreteDomainService, FlextDomainService)
        assert hasattr(service, "__class__")
        assert service.__class__.__name__ == "ConcreteDomainService", (
            f"Expected {'ConcreteDomainService'}, got {service.__class__.__name__}"
        )
