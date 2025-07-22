"""FLEXT Core Tests.

Copyright (c) 2025 Flext. All rights reserved.
SPDX-License-Identifier: MIT

Test suite for FLEXT Core framework.
"""

from __future__ import annotations

import pytest

from flext_core.domain.core import (
    DomainError,
    NotFoundError,
    Repository,
    RepositoryError,
    ValidationError,
)
from flext_core.domain.testing import MockRepository


class TestDomainExceptions:
    """Test domain exception classes."""

    def test_domain_error_creation(self) -> None:
        """Test DomainError can be created."""
        error = DomainError("Test domain error")
        assert str(error) == "Test domain error"
        assert isinstance(error, Exception)

    def test_validation_error_inheritance(self) -> None:
        """Test ValidationError inherits from DomainError."""
        error = ValidationError("Test validation error")
        assert str(error) == "Test validation error"
        assert isinstance(error, DomainError)
        assert isinstance(error, Exception)

    def test_repository_error_inheritance(self) -> None:
        """Test RepositoryError inherits from DomainError."""
        error = RepositoryError("Test repository error")
        assert str(error) == "Test repository error"
        assert isinstance(error, DomainError)
        assert isinstance(error, Exception)

    def test_not_found_error_inheritance(self) -> None:
        """Test NotFoundError inherits from DomainError."""
        error = NotFoundError("Test not found error")
        assert str(error) == "Test not found error"
        assert isinstance(error, DomainError)
        assert isinstance(error, Exception)

    def test_exception_chaining(self) -> None:
        """Test exception chaining works correctly."""
        original_error = ValueError("Original error")

        # Test that the exception is properly raised with cause
        with pytest.raises(DomainError) as exc_info:
            raise DomainError("Domain error") from original_error

        # Verify the exception chain
        assert exc_info.value.__cause__ is original_error
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestRepositoryInterface:
    """Test Repository interface."""

    def test_repository_is_abstract(self) -> None:
        """Test Repository cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Repository()

    def test_repository_abstract_methods(self) -> None:
        """Test Repository has the expected abstract methods."""
        abstract_methods = Repository.__abstractmethods__
        expected_methods = {"save", "find_by_id", "delete", "find_all", "count"}
        assert abstract_methods == expected_methods

    def test_repository_implementation(self) -> None:
        """Test Repository can be properly implemented."""
        # Should be able to instantiate concrete implementation
        repo: MockRepository[str, str] = MockRepository()
        assert repo is not None
        assert hasattr(repo, "save")
        assert hasattr(repo, "find_by_id")
        assert hasattr(repo, "delete")
        assert hasattr(repo, "find_all")
        assert hasattr(repo, "count")


class TestExceptionHierarchy:
    """Test exception hierarchy and behavior."""

    def test_all_domain_errors_are_exceptions(self) -> None:
        """Test all domain errors are proper exceptions."""
        exceptions = [
            DomainError("test"),
            ValidationError("test"),
            RepositoryError("test"),
            NotFoundError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, Exception)
            assert isinstance(exc, DomainError)

    def test_exception_messages(self) -> None:
        """Test exception messages are preserved."""
        test_message = "This is a test error message"

        exceptions = [
            DomainError(test_message),
            ValidationError(test_message),
            RepositoryError(test_message),
            NotFoundError(test_message),
        ]

        for exc in exceptions:
            assert str(exc) == test_message

    def test_exception_raising(self) -> None:
        """Test exceptions can be raised and caught."""
        with pytest.raises(DomainError):
            raise DomainError("test")

        with pytest.raises(ValidationError):
            raise ValidationError("test")

        with pytest.raises(RepositoryError):
            raise RepositoryError("test")

        with pytest.raises(NotFoundError):
            raise NotFoundError("test")

    def test_exception_catching_hierarchy(self) -> None:
        """Test exceptions can be caught by parent classes."""
        # ValidationError should be catchable as DomainError
        with pytest.raises(DomainError):
            raise ValidationError("test")

        # RepositoryError should be catchable as DomainError
        with pytest.raises(DomainError):
            raise RepositoryError("test")

        # NotFoundError should be catchable as DomainError
        with pytest.raises(DomainError):
            raise NotFoundError("test")


class TestModuleStructure:
    """Test module structure and imports."""

    def test_module_exports(self) -> None:
        """Test module exports expected classes."""
        from flext_core.domain import core

        # Check that the module has the expected classes
        assert hasattr(core, "DomainError")
        assert hasattr(core, "ValidationError")
        assert hasattr(core, "RepositoryError")
        assert hasattr(core, "NotFoundError")
        assert hasattr(core, "Repository")

    def test_type_variables(self) -> None:
        """Test type variables are available."""
        from flext_core.domain import core

        assert hasattr(core, "T")
        assert hasattr(core, "ID")
