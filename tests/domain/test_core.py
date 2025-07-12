"""Tests for flext_core.domain.core module."""

import pytest

from flext_core.domain.core import DomainError
from flext_core.domain.core import NotFoundError
from flext_core.domain.core import Repository
from flext_core.domain.core import RepositoryError
from flext_core.domain.core import ValidationError


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
        try:
            msg = "Original error"
            raise ValueError(msg)
        except ValueError as e:
            with pytest.raises(DomainError) as exc_info:
                msg = "Domain error"
                raise DomainError(msg) from e
            assert exc_info.value.__cause__ is e
            assert isinstance(exc_info.value.__cause__, ValueError)


class TestRepositoryInterface:
    """Test Repository interface."""

    def test_repository_is_abstract(self) -> None:
        """Test Repository cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Repository()  # Should fail because it's abstract

    def test_repository_abstract_methods(self) -> None:
        """Test Repository has the expected abstract methods."""
        abstract_methods = Repository.__abstractmethods__
        expected_methods = {"save", "get", "delete", "find_all"}
        assert abstract_methods == expected_methods

    def test_repository_implementation(self) -> None:
        """Test Repository can be properly implemented."""

        class MockRepository(Repository):
            def __init__(self) -> None:
                self.storage = {}

            async def save(self, entity):
                entity_id = getattr(entity, "id", id(entity))
                self.storage[entity_id] = entity
                return entity

            async def get(self, entity_id):
                return self.storage.get(entity_id)

            async def delete(self, entity_id) -> bool:
                if entity_id in self.storage:
                    del self.storage[entity_id]
                    return True
                return False

            async def find_all(self):
                return list(self.storage.values())

        # Should be able to instantiate concrete implementation
        repo = MockRepository()
        assert repo is not None
        assert hasattr(repo, "save")
        assert hasattr(repo, "get")
        assert hasattr(repo, "delete")
        assert hasattr(repo, "find_all")


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
            msg = "test"
            raise DomainError(msg)

        with pytest.raises(ValidationError):
            msg = "test"
            raise ValidationError(msg)

        with pytest.raises(RepositoryError):
            msg = "test"
            raise RepositoryError(msg)

        with pytest.raises(NotFoundError):
            msg = "test"
            raise NotFoundError(msg)

    def test_exception_catching_hierarchy(self) -> None:
        """Test exceptions can be caught by parent classes."""
        # ValidationError should be catchable as DomainError
        with pytest.raises(DomainError):
            msg = "test"
            raise ValidationError(msg)

        # RepositoryError should be catchable as DomainError
        with pytest.raises(DomainError):
            msg = "test"
            raise RepositoryError(msg)

        # NotFoundError should be catchable as DomainError
        with pytest.raises(DomainError):
            msg = "test"
            raise NotFoundError(msg)


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
