"""Test Factories usando Factory Boy para criação de objetos de teste.

Integração massiva com pytest e factory_boy para criação consistente
de objetos de teste em todo o ecosystem flext-core.

Utiliza:
- factory_boy 3.3.3 para criação de objetos
- pytest fixtures para reutilização
- faker para dados realistas
- SOLID principles para extensibilidade
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import factory
from factory import Faker

from flext_core import (
    FlextAggregateRoot,
    FlextConfig,
    FlextEntity,
    FlextFieldCore,
    FlextModel,
    FlextResult,
    FlextValue,
)
from flext_core.constants import FlextEnvironment


class FlextModelFactory(factory.Factory):
    """Factory base para FlextModel objects."""

    class Meta:
        model = FlextModel
        abstract = True

    # Campos base do FlextModel
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))


class FlextValueFactory(factory.Factory):
    """Factory para FlextValue objects."""

    class Meta:
        model = FlextValue

    value = Faker("word")
    metadata = factory.LazyFunction(dict)


class FlextEntityFactory(FlextModelFactory):
    """Factory para FlextEntity objects."""

    class Meta:
        model = FlextEntity
        abstract = True

    id = Faker("uuid4")
    version = 1
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))


class FlextAggregateRootFactory(FlextEntityFactory):
    """Factory para FlextAggregateRoot objects."""

    class Meta:
        model = FlextAggregateRoot
        abstract = True

    # Domain events list vazia por padrão
    domain_events = factory.LazyFunction(list)


class FlextConfigFactory(factory.Factory):
    """Factory para FlextConfig objects."""

    class Meta:
        model = FlextConfig

    environment = factory.Iterator([env.value for env in FlextEnvironment])
    debug = Faker("boolean")
    log_level = factory.Iterator(["DEBUG", "INFO", "WARNING", "ERROR"])
    database_url = Faker("url")
    redis_url = Faker("url")
    secret_key = Faker("password", length=32)


class FlextFieldCoreFactory(factory.Factory):
    """Factory para FlextFieldCore objects."""

    class Meta:
        model = FlextFieldCore

    id = Faker("uuid4")
    name = Faker("word")
    field_type = factory.Iterator(["string", "integer", "boolean", "datetime"])
    required = Faker("boolean")
    description = Faker("sentence")
    default_value = None
    validation_rules = factory.LazyFunction(list)


class FlextResultFactory(factory.Factory):
    """Factory para FlextResult objects."""

    class Meta:
        model = FlextResult
        abstract = True

    @classmethod
    def success(cls, value: Any = None) -> FlextResult[Any]:
        """Cria FlextResult de sucesso."""
        return FlextResult.ok(value or Faker("word").generate())

    @classmethod
    def failure(cls, error: str | None = None) -> FlextResult[Any]:
        """Cria FlextResult de falha."""
        return FlextResult.fail(error or Faker("sentence").generate())

    @classmethod
    def build_success(cls, **kwargs: Any) -> FlextResult[Any]:
        """Build success result with custom value."""
        value = kwargs.get("value", "success_value")
        return FlextResult.ok(value)

    @classmethod
    def build_failure(cls, **kwargs: Any) -> FlextResult[Any]:
        """Build failure result with custom error."""
        error = kwargs.get("error", "test_error")
        return FlextResult.fail(error)


# =============================================================================
# DOMAIN-SPECIFIC FACTORIES
# =============================================================================

class DomainEntityFactory(FlextEntityFactory):
    """Factory para entidades de domínio genéricas."""

    class Meta:
        model = FlextEntity
        abstract = True

    name = Faker("company")
    description = Faker("text", max_nb_chars=200)
    active = True


class UserEntityFactory(DomainEntityFactory):
    """Factory para entidade User específica."""

    class Meta:
        model = FlextEntity
        abstract = True

    email = Faker("email")
    username = Faker("user_name")
    first_name = Faker("first_name")
    last_name = Faker("last_name")
    is_active = True


class OrganizationEntityFactory(DomainEntityFactory):
    """Factory para entidade Organization."""

    class Meta:
        model = FlextEntity
        abstract = True

    name = Faker("company")
    tax_id = Faker("ssn")
    website = Faker("url")
    industry = Faker("job")


# =============================================================================
# CONFIGURATION AND METADATA FACTORIES
# =============================================================================

class MetadataFactory(factory.DictFactory):
    """Factory para metadata dictionaries."""

    source = "test"
    version = "1.0.0"
    created_by = Faker("user_name")
    environment = factory.Iterator([env.value for env in FlextEnvironment])


class ConfigDictFactory(factory.DictFactory):
    """Factory para configuration dictionaries."""

    debug = Faker("boolean")
    timeout = Faker("random_int", min=1, max=300)
    max_retries = Faker("random_int", min=1, max=10)
    batch_size = Faker("random_int", min=10, max=1000)


# =============================================================================
# COLLECTION FACTORIES
# =============================================================================

class EntityListFactory(factory.Factory):
    """Factory para listas de entidades."""

    class Meta:
        model = list

    @classmethod
    def create_batch_entities(cls, size: int = 5) -> list[FlextEntity]:
        """Cria batch de entidades."""
        return [DomainEntityFactory() for _ in range(size)]


class ResultListFactory(factory.Factory):
    """Factory para listas de FlextResult."""

    class Meta:
        model = list

    @classmethod
    def create_mixed_results(cls, success_count: int = 3, failure_count: int = 2) -> list[FlextResult[Any]]:
        """Cria lista mista de sucessos e falhas."""
        results = [FlextResultFactory.success() for _ in range(success_count)]
        results.extend(FlextResultFactory.is_failure() for _ in range(failure_count))
        return results


# =============================================================================
# UTILITY FACTORY FUNCTIONS
# =============================================================================

def create_test_entity(**kwargs: Any) -> FlextEntity:
    """Cria entidade de teste com parâmetros customizados."""
    return DomainEntityFactory(**kwargs)


def create_test_config(**kwargs: Any) -> FlextConfig:
    """Cria config de teste com parâmetros customizados."""
    return FlextConfigFactory(**kwargs)


def create_test_field(**kwargs: Any) -> FlextFieldCore:
    """Cria field de teste com parâmetros customizados."""
    return FlextFieldCoreFactory(**kwargs)


def create_test_metadata(**kwargs: Any) -> dict[str, Any]:
    """Cria metadata de teste."""
    base_metadata = MetadataFactory()
    base_metadata.update(kwargs)
    return base_metadata


# =============================================================================
# BUILDER PATTERN FACTORIES
# =============================================================================

class FlextEntityBuilder:
    """Builder pattern para FlextEntity usando Factory Boy."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def with_id(self, entity_id: str) -> FlextEntityBuilder:
        """Define ID da entidade."""
        self._data["id"] = entity_id
        return self

    def with_name(self, name: str) -> FlextEntityBuilder:
        """Define nome da entidade."""
        self._data["name"] = name
        return self

    def with_metadata(self, **metadata: Any) -> FlextEntityBuilder:
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

    def build(self) -> FlextEntity:
        """Constrói a entidade."""
        return DomainEntityFactory(**self._data)


class FlextResultBuilder:
    """Builder pattern para FlextResult."""

    def __init__(self) -> None:
        self._success = True
        self._value: Any = None
        self._error: str | None = None

    def success(self, value: Any = None) -> FlextResultBuilder:
        """Configura como sucesso."""
        self._success = True
        self._value = value
        return self

    def failure(self, error: str) -> FlextResultBuilder:
        """Configura como falha."""
        self._success = False
        self._error = error
        return self

    def build(self) -> FlextResult[Any]:
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
