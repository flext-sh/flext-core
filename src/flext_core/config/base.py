"""Enhanced configuration system with declarative patterns and dependency injection.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Self
from typing import TypeVar

from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import SettingsConfigDict

from flext_core.domain.constants import ConfigDefaults
from flext_core.domain.constants import FlextFramework
from flext_core.domain.core import DomainError
from flext_core.domain.pydantic_base import DomainBaseModel
from flext_core.domain.types import EnvironmentLiteral
from flext_core.domain.types import ProjectName
from flext_core.domain.types import Version

# Type variables
T = TypeVar("T")


class ConfigurationError(DomainError):
    """Configuration-related errors."""


class BaseConfig(DomainBaseModel):
    """Enhanced base configuration model with declarative patterns.

    REFACTORED: Uses ConfigurationMixin for maximum code reduction.
    All configuration models should inherit from this class.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        frozen=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.

        """
        return self.model_dump()

    def get_subsection(self, prefix: str) -> dict[str, object]:
        """Get a subsection of the configuration.

        Arguments:
            prefix: The prefix to get the subsection for.

        Returns:
            A dictionary of the subsection.

        """
        data = self.model_dump()
        return {k[len(prefix) :]: v for k, v in data.items() if k.startswith(prefix)}


class BaseSettings(PydanticBaseSettings):
    """Enhanced base settings class with declarative configuration support.

    REFACTORED: Uses ConfigurationMixin for maximum code reduction.
    Extends pydantic-settings BaseSettings with FLEXT-specific defaults
    and enhanced functionality for dependency injection.
    """

    model_config = SettingsConfigDict(
        # Environment variable settings
        env_prefix=ConfigDefaults.ENV_PREFIX,  # Default prefix for all FLEXT projects
        env_file=".env",  # Load from .env file
        env_file_encoding=ConfigDefaults.DEFAULT_ENCODING,
        env_nested_delimiter=ConfigDefaults.ENV_DELIMITER,  # For nested configs
        case_sensitive=False,
        # Pydantic settings
        extra="ignore",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    # Project identification using modern typing
    project_name: ProjectName = Field(default="flext", description="Project name")
    project_version: Version = Field(
        default=FlextFramework.VERSION,
        description="Project version",
    )

    # Environment settings using typed literals
    environment: EnvironmentLiteral = Field(
        default="development",
        description="Environment name",
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    @classmethod
    def get_env_prefix(cls) -> str:
        """Get the environment prefix.

        Returns:
            The environment prefix.

        """
        return cls.model_config.get("env_prefix", "FLEXT_")

    @classmethod
    def from_env(cls, env_file: str | None = None) -> Self:
        """Create a settings instance from the environment.

        Arguments:
            env_file: The environment file to load.

        Raises:
            ConfigurationError: If the settings are invalid.

        Returns:
            The settings instance.

        """
        try:
            # Create instance with custom env_file if provided
            if env_file:
                # Simple manual env file parsing (avoiding python-dotenv dependency)
                original_env: dict[str, str | None] = {}
                env_data: dict[str, str] = {}

                # Read and parse the env file manually
                env_path = Path(env_file)
                if env_path.exists():
                    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                        env_line = raw_line.strip()
                        if (
                            env_line
                            and not env_line.startswith("#")
                            and "=" in env_line
                        ):
                            key, value = env_line.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            env_data[key] = value

                            # Store original value for restoration
                            original_env[key] = os.environ.get(key)
                            os.environ[key] = value

                try:
                    return cls()
                finally:
                    # Restore original environment
                    for key, original_value in original_env.items():
                        if original_value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = original_value
            else:
                return cls()
        except ValidationError as e:
            msg = f"Invalid settings: {e}"
            raise ConfigurationError(msg) from e

    def to_env_dict(self) -> dict[str, str]:
        """Convert the settings to a dictionary of environment variables.

        Returns:
            A dictionary of environment variables.

        """
        prefix = self.get_env_prefix()
        env_dict = {}

        for field_name, value in self.model_dump().items():
            if value is not None:
                env_key = f"{prefix}{field_name.upper()}"
                env_dict[env_key] = str(value)

        return env_dict

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary.

        Returns:
            Dictionary representation of settings.

        """
        return self.model_dump()

    def configure_dependencies(self, container: "DIContainer") -> None:
        """Configure the dependencies.

        Arguments:
            container: The dependency injection container.

        """
        # Register this settings instance
        container.register(type(self), self)

    def get_subsection(self, prefix: str) -> dict[str, object]:
        """Get a subsection of the settings.

        Arguments:
            prefix: The prefix to get the subsection for.

        Returns:
            A dictionary of the subsection.

        """
        data = self.model_dump()
        return {k[len(prefix) :]: v for k, v in data.items() if k.startswith(prefix)}


# Dependency Injection Infrastructure
class DIContainer:
    """Simple dependency injection container for FLEXT projects."""

    def __init__(self) -> None:
        """Initialize the dependency injection container."""
        self._services: dict[type, object] = {}
        self._factories: dict[type, Callable[[], Any]] = {}
        self._singletons: dict[type, object] = {}
        self._resolving: set[type] = set()  # Track what we're currently resolving

    def register(self, service_type: type[T], instance: T) -> None:
        """Register a service.

        Arguments:
            service_type: The type of the service.
            instance: The instance of the service.

        """
        self._services[service_type] = instance

    def register_factory(
        self,
        service_type: type[T],
        factory: Callable[..., T],
    ) -> None:
        """Register a factory.

        Arguments:
            service_type: The type of the service.
            factory: The factory function.

        """
        self._factories[service_type] = factory

    def register_singleton(
        self,
        service_type: type[T],
        factory: Callable[..., T],
    ) -> None:
        """Register a singleton.

        Arguments:
            service_type: The type of the service.
            factory: The factory function.

        """
        self._factories[service_type] = factory
        # Mark as singleton
        self._singletons[service_type] = None

    def resolve(self, service_type: type[T]) -> T:
        """Resolve a service.

        Arguments:
            service_type: The type of the service.

        Returns:
            The instance of the service.

        """
        # Check for recursion
        if service_type in self._resolving:
            msg = f"Circular dependency detected for {service_type}"
            raise ConfigurationError(msg)

        # Check if already registered
        if service_type in self._services:
            return self._services[service_type]  # type: ignore[return-value]

        # Check if singleton already created
        if (
            service_type in self._singletons
            and self._singletons[service_type] is not None
        ):
            return self._singletons[service_type]  # type: ignore[return-value]

        # Mark as resolving
        self._resolving.add(service_type)
        try:
            # Check if factory exists
            if service_type in self._factories:
                factory = self._factories[service_type]
                instance = self._create_with_injection(factory)

                # Store singleton
                if service_type in self._singletons:
                    self._singletons[service_type] = instance

                return instance  # type: ignore[no-any-return]

            # Try to create automatically
            return self._create_with_injection(service_type)
        finally:
            # Remove from resolving set
            self._resolving.discard(service_type)

    def _create_with_injection(self, factory_or_type: type[T] | Callable[..., T]) -> T:
        """Create a service with injection.

        Arguments:
            factory_or_type: The factory or type to create.

        Raises:
            ConfigurationError: If the service cannot be resolved.

        Returns:
            The instance of the service.

        """
        if inspect.isclass(factory_or_type):
            # It's a class, get its __init__ method
            init_signature = inspect.signature(factory_or_type.__init__)
            params = {}

            for param_name, param in init_signature.parameters.items():
                if param_name == "self":
                    continue

                if param.annotation != inspect.Parameter.empty:
                    # Check if we have this dependency registered
                    if (
                        param.annotation in self._services
                        or param.annotation in self._factories
                    ):
                        # Try to resolve the dependency
                        try:
                            params[param_name] = self.resolve(param.annotation)
                        except (TypeError, ValueError, KeyError, RecursionError):
                            # If can't resolve and has default, use default
                            if param.default != inspect.Parameter.empty:
                                params[param_name] = param.default
                            else:
                                msg = (
                                    f"Cannot resolve dependency {param.annotation} "
                                    f"for {factory_or_type}"
                                )
                                raise ConfigurationError(msg) from None
                    # No registration found, use default if available
                    elif param.default != inspect.Parameter.empty:
                        params[param_name] = param.default
                    else:
                        # Try to auto-create if it's a class
                        try:
                            if inspect.isclass(param.annotation):
                                params[param_name] = self.resolve(param.annotation)
                            else:
                                msg = (
                                    f"Cannot resolve dependency {param.annotation} "
                                    f"for {factory_or_type}"
                                )
                                raise ConfigurationError(msg) from None
                        except (TypeError, ValueError, KeyError, RecursionError):
                            msg = (
                                f"Cannot resolve dependency {param.annotation} "
                                f"for {factory_or_type}"
                            )
                            raise ConfigurationError(msg) from None
                # Parameter has no annotation, use default if available
                elif param.default != inspect.Parameter.empty:
                    params[param_name] = param.default

            instance: T = factory_or_type(**params)
            return instance
        # It's a factory function
        signature = inspect.signature(factory_or_type)
        params = {}

        for param_name, param in signature.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                # Check if we have this dependency registered
                if (
                    param.annotation in self._services
                    or param.annotation in self._factories
                ):
                    try:
                        params[param_name] = self.resolve(param.annotation)
                    except (TypeError, ValueError, KeyError, RecursionError):
                        if param.default != inspect.Parameter.empty:
                            params[param_name] = param.default
                        else:
                            msg = f"Cannot resolve dependency {param.annotation} for factory"
                            raise ConfigurationError(msg) from None
                # No registration found, use default if available
                elif param.default != inspect.Parameter.empty:
                    params[param_name] = param.default
                else:
                    msg = f"Cannot resolve dependency {param.annotation} for factory"
                    raise ConfigurationError(msg) from None
            # Parameter has no annotation, use default if available
            elif param.default != inspect.Parameter.empty:
                params[param_name] = param.default

        result: T = factory_or_type(**params)
        return result


# Global DI container instance
_container: DIContainer | None = None


def get_container() -> DIContainer:
    """Get the dependency injection container.

    Returns:
        The dependency injection container.

    """
    global _container  # noqa: PLW0603
    if _container is None:
        _container = DIContainer()
    return _container


def configure_container(container: DIContainer | None = None) -> DIContainer:
    """Configure the dependency injection container.

    Arguments:
        container: The dependency injection container.

    Returns:
        The dependency injection container.

    """
    global _container  # noqa: PLW0603
    if container is None:
        container = DIContainer()
    _container = container
    return container


# Decorators for dependency injection
def injectable[T](service_type: type[T] | None = None) -> Callable[[type[T]], type[T]]:
    """Injectable decorator.

    Arguments:
        service_type: The type of the service.

    Returns:
        The decorator.

    """

    def decorator(cls: type[T]) -> type[T]:
        container = get_container()
        if service_type:
            container.register_factory(service_type, cls)
        else:
            container.register_factory(cls, cls)
        return cls

    return decorator


def singleton[T](service_type: type[T] | None = None) -> Callable[[type[T]], type[T]]:
    """Singleton decorator.

    Arguments:
        service_type: The type of the service.

    Returns:
        The decorator.

    """

    def decorator(cls: type[T]) -> type[T]:
        container = get_container()
        if service_type:
            container.register_singleton(service_type, cls)
        else:
            container.register_singleton(cls, cls)
        return cls

    return decorator


# Declarative configuration helper
class ConfigSection:
    """Declarative configuration section descriptor."""

    def __init__(self, config_class: type[BaseConfig], prefix: str = "") -> None:
        """Initialize the configuration section.

        Arguments:
            config_class: The configuration class.
            prefix: The prefix of the configuration.

        """
        self.config_class = config_class
        self.prefix = prefix

    def __get__(
        self,
        instance: BaseConfig | None,
        owner: type | None = None,
    ) -> BaseConfig | object:
        """Get the configuration section.

        Arguments:
            instance: The instance of the configuration.
            owner: The owner of the configuration.

        Returns:
            The configuration section.

        """
        if instance is None:
            return self

        # Extract subsection from settings
        subsection_data = instance.get_subsection(self.prefix)
        return self.config_class.model_validate(subsection_data)

    def __set__(self, instance: BaseConfig, value: BaseConfig) -> None:
        """Set the configuration section.

        Arguments:
            instance: The instance of the configuration.
            value: The value of the configuration.

        Raises:
            TypeError: If the value is not an instance of the configuration class.

        """
        if not isinstance(value, self.config_class):
            msg = f"Value must be instance of {self.config_class}"
            raise TypeError(msg)
        # Note: This is a simplified implementation
        # In practice, you might want to update the underlying settings


# Type variable for generic configuration
TConfig = TypeVar("TConfig", bound=BaseConfig)
TSettings = TypeVar("TSettings", bound=BaseSettings)


# Updated convenience functions
def get_config[TConfig: BaseConfig](
    config_class: type[TConfig],
    data: dict[str, object] | None = None,
) -> TConfig:
    """Get the configuration.

    Arguments:
        config_class: The configuration class.
        data: The data to get the configuration from.

    Returns:
        The configuration.

    """
    return config_class.model_validate(data or {})


def get_settings[TSettings: BaseSettings](
    settings_class: type[TSettings],
    env_file: str | None = None,
) -> TSettings:
    """Get the settings.

    Arguments:
        settings_class: The settings class.
        env_file: The environment file to load.

    Returns:
        The settings.

    """
    return settings_class.from_env(env_file)
