"""Test basic imports and module structure."""


def test_flext_core_imports() -> None:
    """Test that flext_core can be imported successfully."""
    import flext_core

    assert flext_core.__version__ == "0.7.0"
    assert hasattr(flext_core, "BaseConfig")
    assert hasattr(flext_core, "DomainEntity")
    assert hasattr(flext_core, "Pipeline")


def test_configuration_imports() -> None:
    """Test configuration system imports."""
    from flext_core import BaseConfig
    from flext_core import BaseSettings
    from flext_core import ConfigSection
    from flext_core import ConfigurationError
    from flext_core import DIContainer
    from flext_core import get_config
    from flext_core import get_settings

    assert BaseConfig is not None
    assert BaseSettings is not None
    assert ConfigSection is not None
    assert ConfigurationError is not None
    assert DIContainer is not None
    assert callable(get_config)
    assert callable(get_settings)


def test_domain_imports() -> None:
    """Test domain layer imports."""
    from flext_core import DomainEntity
    from flext_core import DomainError
    from flext_core import DomainValueObject
    from flext_core import Pipeline
    from flext_core import PipelineId
    from flext_core import ServiceResult

    assert DomainEntity is not None
    assert DomainValueObject is not None
    assert DomainError is not None
    assert Pipeline is not None
    assert PipelineId is not None
    assert ServiceResult is not None


def test_shared_models_imports() -> None:
    """Test shared models imports."""
    from flext_core import AuthToken
    from flext_core import ComponentHealth
    from flext_core import DatabaseConfig
    from flext_core import HealthStatus
    from flext_core import LDAPEntry
    from flext_core import OperationStatus
    from flext_core import PluginMetadata
    from flext_core import SystemHealth
    from flext_core import UserInfo

    assert AuthToken is not None
    assert ComponentHealth is not None
    assert DatabaseConfig is not None
    assert HealthStatus is not None
    assert LDAPEntry is not None
    assert OperationStatus is not None
    assert PluginMetadata is not None
    assert SystemHealth is not None
    assert UserInfo is not None


def test_types_imports() -> None:
    """Test types imports."""
    from flext_core import EntityId
    from flext_core import EntityStatus
    from flext_core import Environment
    from flext_core import PluginId
    from flext_core import ProjectName
    from flext_core import ResultStatus
    from flext_core import Status
    from flext_core import UserId
    from flext_core import Version

    assert EntityId is not None
    assert EntityStatus is not None
    assert Environment is not None
    assert PluginId is not None
    assert ProjectName is not None
    assert ResultStatus is not None
    assert Status is not None
    assert UserId is not None
    assert Version is not None


def test_constants_imports() -> None:
    """Test constants imports."""
    from flext_core import EntityStatuses
    from flext_core import Environments
    from flext_core import ErrorMessages
    from flext_core import ExecutionStatuses
    from flext_core import FlextFramework
    from flext_core import LogLevels
    from flext_core import PluginTypes
    from flext_core import ResultStatuses
    from flext_core import SuccessMessages

    assert EntityStatuses is not None
    assert Environments is not None
    assert ErrorMessages is not None
    assert ExecutionStatuses is not None
    assert FlextFramework is not None
    assert LogLevels is not None
    assert PluginTypes is not None
    assert ResultStatuses is not None
    assert SuccessMessages is not None
