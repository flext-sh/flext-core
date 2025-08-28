# ruff: noqa: ANN401, S311
"""Test Factories para criação de objetos de teste compatível com flext-core refatorado.

Fornece factories simplificadas para criação consistente de objetos de teste
sem dependências externas, utilizando apenas a biblioteca padrão Python.

Compatível com:
- flext-core nova estrutura (FlextResult.value, FlextTypes)
- Python 3.13+ type system
- Pytest fixtures nativas
- SOLID principles para extensibilidade
"""

from __future__ import annotations

import random
import string
import uuid
from datetime import UTC, datetime
from typing import override

from flext_core import (
    FlextAggregates,
    FlextConfig,
    FlextConstants,
    FlextEntity,
    FlextFields,
    FlextModels,
    FlextResult,
)


def _generate_fake_word() -> str:
    """Generate a fake word for testing."""
    return "".join(random.choices(string.ascii_lowercase, k=random.randint(4, 12)))


def _generate_fake_sentence() -> str:
    """Generate a fake sentence for testing."""
    words = [_generate_fake_word() for _ in range(random.randint(3, 8))]
    return " ".join(words).capitalize() + "."


def _generate_fake_email() -> str:
    """Generate a fake email for testing."""
    user = _generate_fake_word()
    domain = _generate_fake_word()
    return f"{user}@{domain}.com"


def _generate_fake_url() -> str:
    """Generate a fake URL for testing."""
    domain = _generate_fake_word()
    return f"https://{domain}.com"


def _generate_fake_company() -> str:
    """Generate a fake company name for testing."""
    return _generate_fake_word().capitalize() + " Corp"


class FlextModelFactory:
    """Factory base para FlextModel objects sem dependências externas."""

    @classmethod
    def create(cls, **kwargs: object) -> FlextModel:
        """Create FlextModel instance with default values."""
        defaults = {
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        defaults.update(kwargs)
        return FlextModels(**defaults)


class TestValueObject(FlextModels.Value):
    """Concrete FlextValue implementation for testing."""

    value: object = None

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Test implementation of business rules validation."""
        return FlextResult[None].ok(None)


class FlextValueFactory:
    """Factory para FlextValue objects."""

    @classmethod
    def create(cls, **kwargs: object) -> TestValueObject:
        """Create FlextValue instance with default values."""
        defaults = {
            "value": _generate_fake_word(),
        }
        defaults.update(kwargs)
        return TestValueObject(**defaults)


class TestEntity(FlextEntity):
    """Concrete FlextEntity implementation for testing."""

    name: str = "Test Entity"

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Test implementation of business rules validation."""
        return FlextResult[None].ok(None)


class FlextEntityFactory:
    """Factory para FlextEntity objects."""

    @classmethod
    def create(cls, **kwargs: object) -> TestEntity:
        """Create FlextEntity instance with default values."""
        # FlextEntity will provide defaults for id, version, timestamps, etc.
        defaults = {
            "name": "Test Entity",
        }
        defaults.update(kwargs)
        # Create with auto-generated ID if not provided
        if "id" not in defaults:
            defaults["id"] = f"test_entity_{uuid.uuid4().hex[:8]}"
        return TestEntity(**defaults)


class TestAggregateRoot(FlextAggregates):
    """Concrete FlextAggregates implementation for testing."""

    name: str = "Test Aggregate"

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Test implementation of business rules validation."""
        return FlextResult[None].ok(None)


class FlextAggregateRootFactory:
    """Factory para FlextAggregates objects."""

    @classmethod
    def create(cls, **kwargs: object) -> TestAggregateRoot:
        """Create FlextAggregates instance with default values."""
        defaults = {
            "name": "Test Aggregate",
        }
        defaults.update(kwargs)
        return TestAggregateRoot(**defaults)


class FlextConfigFactory:
    """Factory para FlextConfig objects."""

    @classmethod
    def create(cls, **kwargs: object) -> FlextConfig:
        """Create FlextConfig instance with default values."""
        defaults = {
            "name": "test-flext",
            "version": "1.0.0",
            "description": "Test FLEXT configuration",
            "environment": random.choice([
                env.value for env in FlextConstants.Config.ConfigEnvironment
            ]),
            "debug": random.choice([True, False]),
            "log_level": random.choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
            "timeout": random.randint(5, 60),
            "retries": random.randint(1, 5),
            "page_size": random.randint(10, 500),
            "enable_caching": random.choice([True, False]),
            "enable_metrics": random.choice([True, False]),
            "enable_tracing": random.choice([True, False]),
        }
        defaults.update(kwargs)
        return FlextConfig(**defaults)


class FlextFieldCoreFactory:
    """Factory para FlextFieldCore objects."""

    @classmethod
    def create(cls, **kwargs: object) -> object:
        """Create FlextFields.Core field instance with default values."""
        defaults = {
            "name": _generate_fake_word(),
            "field_type": random.choice(["string", "integer", "boolean", "datetime"]),
            "required": random.choice([True, False]),
            "description": _generate_fake_sentence(),
            "default_value": None,
        }
        defaults.update(kwargs)
        # Use proper FLEXT hierarchical pattern
        field_name = defaults.get("name", "test_field")
        return FlextFields.Core.StringField(str(field_name))


class FlextResultFactory:
    """Factory para FlextResult objects."""

    @classmethod
    def success(cls, value: object = None) -> FlextResult[object]:
        """Cria FlextResult de sucesso."""
        return FlextResult.ok(value or _generate_fake_word())

    @classmethod
    def failure(cls, error: str | None = None) -> FlextResult[object]:
        """Cria FlextResult de falha."""
        return FlextResult.fail(error or _generate_fake_sentence())

    @classmethod
    def build_success(cls, **kwargs: object) -> FlextResult[object]:
        """Build success result with custom value."""
        value = kwargs.get("value", "success_value")
        return FlextResult.ok(value)

    @classmethod
    def build_failure(cls, **kwargs: object) -> FlextResult[object]:
        """Build failure result with custom error."""
        error = kwargs.get("error", "test_error")
        return FlextResult.fail(str(error))


# =============================================================================
# DOMAIN-SPECIFIC FACTORIES
# =============================================================================


class DomainEntity(FlextEntity):
    """Concrete domain entity for testing."""

    name: str = "Test Domain Entity"
    description: str = "Test description"
    active: bool = True

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Test implementation of business rules validation."""
        return FlextResult[None].ok(None)


class DomainEntityFactory:
    """Factory para entidades de domínio genéricas."""

    @classmethod
    def create(cls, **kwargs: object) -> DomainEntity:
        """Create domain entity instance with default values."""
        defaults = {
            "id": str(uuid.uuid4()),
            "version": 1,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "name": _generate_fake_company(),
            "description": _generate_fake_sentence(),
            "active": True,
        }
        defaults.update(kwargs)
        return DomainEntity(**defaults)


class UserEntity(FlextEntity):
    """Concrete user entity for testing."""

    email: str = "test@example.com"
    username: str = "testuser"
    first_name: str = "Test"
    last_name: str = "User"
    is_active: bool = True

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Test implementation of business rules validation."""
        return FlextResult[None].ok(None)


class UserEntityFactory:
    """Factory para entidade User específica."""

    @classmethod
    def create(cls, **kwargs: object) -> UserEntity:
        """Create user entity instance with default values."""
        defaults = {
            "id": str(uuid.uuid4()),
            "version": 1,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "email": _generate_fake_email(),
            "username": _generate_fake_word(),
            "first_name": _generate_fake_word().capitalize(),
            "last_name": _generate_fake_word().capitalize(),
            "is_active": True,
        }
        defaults.update(kwargs)
        return UserEntity(**defaults)


class OrganizationEntity(FlextEntity):
    """Concrete organization entity for testing."""

    name: str = "Test Organization"
    tax_id: str = "123456789"
    website: str = "https://test.com"
    industry: str = "Technology"

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Test implementation of business rules validation."""
        return FlextResult[None].ok(None)


class OrganizationEntityFactory:
    """Factory para entidade Organization."""

    @classmethod
    def create(cls, **kwargs: object) -> OrganizationEntity:
        """Create organization entity instance with default values."""
        defaults = {
            "id": str(uuid.uuid4()),
            "version": 1,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "name": _generate_fake_company(),
            "tax_id": "".join(random.choices(string.digits, k=9)),
            "website": _generate_fake_url(),
            "industry": _generate_fake_word().capitalize(),
        }
        defaults.update(kwargs)
        return OrganizationEntity(**defaults)


# =============================================================================
# CONFIGURATION AND METADATA FACTORIES
# =============================================================================


class MetadataFactory:
    """Factory para metadata dictionaries."""

    @classmethod
    def create(cls, **kwargs: object) -> dict[str, object]:
        """Create metadata dictionary with default values."""
        defaults: dict[str, object] = {
            "source": "test",
            "version": "1.0.0",
            "created_by": _generate_fake_word(),
            "environment": random.choice([
                env.value for env in FlextConstants.Config.ConfigEnvironment
            ]),
        }
        defaults.update(kwargs)
        return defaults


class ConfigDictFactory:
    """Factory para configuration dictionaries."""

    @classmethod
    def create(cls, **kwargs: object) -> dict[str, object]:
        """Create config dictionary with default values."""
        defaults: dict[str, object] = {
            "debug": random.choice([True, False]),
            "timeout": random.randint(1, 300),
            "max_retries": random.randint(1, 10),
            "batch_size": random.randint(10, 1000),
        }
        defaults.update(kwargs)
        return defaults


# =============================================================================
# COLLECTION FACTORIES
# =============================================================================


class EntityListFactory:
    """Factory para listas de entidades."""

    @classmethod
    def create_batch_entities(cls, size: int = 5) -> list[DomainEntity]:
        """Cria batch de entidades."""
        return [DomainEntityFactory.create() for _ in range(size)]


class ResultListFactory:
    """Factory para listas de FlextResult."""

    @classmethod
    def create_mixed_results(
        cls, success_count: int = 3, failure_count: int = 2
    ) -> list[FlextResult[object]]:
        """Cria lista mista de sucessos e falhas."""
        results = [FlextResultFactory.success() for _ in range(success_count)]
        results.extend(FlextResultFactory.failure() for _ in range(failure_count))
        return results


# =============================================================================
# UTILITY FACTORY FUNCTIONS
# =============================================================================


def create_test_entity(**kwargs: object) -> DomainEntity:
    """Cria entidade de teste com parâmetros customizados."""
    return DomainEntityFactory.create(**kwargs)


def create_test_config(**kwargs: object) -> FlextConfig:
    """Cria config de teste com parâmetros customizados."""
    return FlextConfigFactory.create(**kwargs)


def create_test_field(**kwargs: object) -> object:
    """Cria field de teste com parâmetros customizados."""
    return FlextFieldCoreFactory.create(**kwargs)


def create_test_metadata(**kwargs: object) -> dict[str, object]:
    """Cria metadata de teste."""
    base_metadata = MetadataFactory.create()
    base_metadata.update(kwargs)
    return base_metadata


# =============================================================================
# BUILDER PATTERN FACTORIES
# =============================================================================


class FlextEntityBuilder:
    """Builder pattern para FlextEntity usando Factory Boy."""

    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def with_id(self, entity_id: str) -> FlextEntityBuilder:
        """Define ID da entidade."""
        self._data["id"] = entity_id
        return self

    def with_name(self, name: str) -> FlextEntityBuilder:
        """Define nome da entidade."""
        self._data["name"] = name
        return self

    def with_metadata(self, **metadata: object) -> FlextEntityBuilder:
        """Define metadata da entidade."""
        self._data.update(metadata)
        return self

    def active(self) -> FlextEntityBuilder:
        """Marca entidade como ativa."""
        self._data["active"] = True
        return self

    def inactive(self) -> FlextEntityBuilder:
        """Marca entidade como inativa."""
        self._data["active"] = False
        return self

    def build(self) -> DomainEntity:
        """Constrói a entidade."""
        return DomainEntityFactory.create(**self._data)


class FlextResultBuilder:
    """Builder pattern para FlextResult."""

    def __init__(self) -> None:
        self._success = True
        self._value: object = None
        self._error: str | None = None

    def success(self, value: object = None) -> FlextResultBuilder:
        """Configura como sucesso."""
        self._success = True
        self._value = value
        return self

    def failure(self, error: str) -> FlextResultBuilder:
        """Configura como falha."""
        self._success = False
        self._error = error
        return self

    def build(self) -> FlextResult[object]:
        """Constrói o FlextResult."""
        if self._success:
            return FlextResult.ok(self._value)
        return FlextResult.fail(self._error or "Test error")


# =============================================================================
# PYTEST INTEGRATION FACTORIES
# =============================================================================


def pytest_entity_factory() -> type[DomainEntityFactory]:
    """Factory para uso com pytest fixtures."""
    return DomainEntityFactory


def pytest_config_factory() -> type[FlextConfigFactory]:
    """Factory para uso com pytest fixtures."""
    return FlextConfigFactory


def pytest_result_factory() -> type[FlextResultFactory]:
    """Factory para uso com pytest fixtures."""
    return FlextResultFactory


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ConfigDictFactory",
    # Domain factories
    "DomainEntityFactory",
    # Collection factories
    "EntityListFactory",
    "FlextAggregateRootFactory",
    "FlextConfigFactory",
    # Builders
    "FlextEntityBuilder",
    "FlextEntityFactory",
    "FlextFieldCoreFactory",
    # Base factories
    "FlextModelFactory",
    "FlextResultBuilder",
    "FlextResultFactory",
    "FlextValueFactory",
    # Metadata factories
    "MetadataFactory",
    "OrganizationEntityFactory",
    "ResultListFactory",
    "UserEntityFactory",
    "create_test_config",
    # Utility functions
    "create_test_entity",
    "create_test_field",
    "create_test_metadata",
    "pytest_config_factory",
    # Pytest integration
    "pytest_entity_factory",
    "pytest_result_factory",
]
