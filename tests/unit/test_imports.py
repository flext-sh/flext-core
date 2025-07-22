"""Test basic imports and module structure.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides tests to ensure that the flext_core package
can be imported successfully and that the module structure is correct.
"""

from __future__ import annotations


def test_flext_core_imports() -> None:
    """Test that flext_core can be imported successfully."""
    import flext_core

    assert flext_core.__version__ == "0.7.0"
    assert hasattr(flext_core, "BaseConfig")
    assert hasattr(flext_core, "DomainEntity")
    assert hasattr(flext_core, "ServiceResult")


def test_configuration_imports() -> None:
    """Test configuration system imports."""
    from flext_core import (
        BaseConfig,
        BaseSettings,
        DIContainer,
        configure_container,
        get_container,
        injectable,
        singleton,
    )

    assert BaseConfig is not None
    assert BaseSettings is not None
    assert DIContainer is not None
    assert callable(configure_container)
    assert callable(get_container)
    assert callable(injectable)
    assert callable(singleton)


def test_domain_imports() -> None:
    """Test domain layer imports."""
    from flext_core import (
        DomainAggregateRoot,
        DomainBaseModel,
        DomainEntity,
        DomainError,
        DomainEvent,
        DomainValueObject,
        ServiceResult,
        ValidationError,
    )

    assert DomainAggregateRoot is not None
    assert DomainBaseModel is not None
    assert DomainEntity is not None
    assert DomainValueObject is not None
    assert DomainError is not None
    assert DomainEvent is not None
    assert ServiceResult is not None
    assert ValidationError is not None


def test_pydantic_imports() -> None:
    """Test Pydantic-related imports."""
    from flext_core import (
        BaseModel,
        Field,
        PydanticBaseSettings,
    )

    assert BaseModel is not None
    assert Field is not None
    assert PydanticBaseSettings is not None


def test_submodule_imports() -> None:
    """Test that submodules can be imported directly."""
    # Test domain types imports
    from flext_core.domain.shared_types import EntityStatus, ExecutionStatus
    from flext_core.domain.shared_types import ServiceResult
    assert EntityStatus is not None
    assert ExecutionStatus is not None
    assert ServiceResult is not None

    # Test domain constants
    from flext_core.domain.constants import FlextFramework

    assert FlextFramework is not None


def test_direct_module_imports() -> None:
    """Test direct module imports for comprehensive functionality."""
    # Test configuration validators
    from flext_core.config.validators import (
        validate_port,
        validate_timeout,
        validate_url,
    )

    assert callable(validate_port)
    assert callable(validate_timeout)
    assert callable(validate_url)

    # Test infrastructure components
    from flext_core.infrastructure.memory import InMemoryRepository

    assert InMemoryRepository is not None
