"""Protocol metaclass and protocol-enabled Pydantic base models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel
from pydantic._internal._model_construction import (
    ModelMetaclass as _TypeCheckModelMeta,
)
from pydantic_settings import BaseSettings

from flext_core import T, t
from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.introspection import METACLASS_STRICT, ProtocolIntrospection

if TYPE_CHECKING:

    class _CombinedModelMeta(_TypeCheckModelMeta):
        """TYPE_CHECKING stub: metaclass chain for mypy resolution."""

else:

    def _build_combined_model_meta() -> type:
        return type("_CombinedModelMeta", (type(BaseModel), type(Protocol)), {})

    _CombinedModelMeta: type = _build_combined_model_meta()


class ProtocolModelMeta(_CombinedModelMeta):
    """Metaclass combining Pydantic with Protocol structural typing.

    This metaclass inherits from a dynamically-created combined metaclass
    that includes both Pydantic's ModelMetaclass AND Protocol's _ProtocolMeta.
    This allows classes using this metaclass to inherit from Protocol
    subclasses without metaclass conflicts.

    The key insight is to separate Protocol types from real bases,
    create the class with only model bases (avoiding metaclass conflict),
    then validate and store protocol information for runtime checking.

    Usage:
        class MyModel(p.ProtocolModel, p.Domain.Entity):
            name: str
            value: int
    """

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: Mapping[
            str,
            t.Container | BaseModel | type | Callable[..., t.Container | BaseModel],
        ],
        **_kwargs: t.Scalar,
    ) -> type:
        """Create a new class with protocol validation.

        Args:
            name: The class name.
            bases: Tuple of base classes (may include protocols).
            namespace: The class namespace dictionary.
            **_kwargs: Additional keyword arguments for metaclass.

        Returns:
            The newly created class with protocols validated.

        """
        protocols, model_bases = ProtocolIntrospection.partition_protocol_bases(bases)
        if not model_bases:
            model_bases = [BaseModel]
        built_cls: type = super().__new__(
            cls, name, tuple(model_bases), dict(namespace)
        )
        setattr(built_cls, "__protocols__", tuple(protocols))
        if METACLASS_STRICT:
            for protocol in protocols:
                FlextProtocolsBase.validate_protocol_compliance(
                    built_cls, protocol, name
                )
        return built_cls


class ProtocolModel(metaclass=ProtocolModelMeta):
    """Base class for Pydantic models that implement protocols.

    Enables natural multi-inheritance with protocols without metaclass
    conflicts. Protocol compliance is validated at class definition time.

    Usage:
        class MyEntity(p.ProtocolModel, p.Domain.Entity):
            name: str
            value: int

        # Check protocols at runtime
        entity = MyEntity(name="test", value=42)
        assert entity.implements_protocol(p.Domain.Entity)
        assert entity.get_protocols() == (p.Domain.Entity,)
    """

    @classmethod
    def get_protocols(cls) -> tuple[type, ...]:
        """Return all protocols this class implements.

        Returns:
            Tuple of protocol types.

        """
        return ProtocolIntrospection.get_class_protocols(cls)

    def implements_protocol(self, protocol: type) -> bool:
        """Check if this instance implements a protocol.

        Args:
            protocol: The protocol type to check.

        Returns:
            True if this instance implements the protocol.

        """
        return ProtocolIntrospection.check_implements_protocol(self, protocol)


class ProtocolSettings(BaseSettings, metaclass=ProtocolModelMeta):
    """Base class for Pydantic Settings that implement protocols.

    Extends the ProtocolModel pattern to BaseSettings, enabling
    environment variable loading alongside protocol compliance.

    Usage:
        class MySettings(p.ProtocolSettings, p.Configuration.Config):
            app_name: str = Field(default="myapp")
            debug: bool = Field(default=False)

            model_config = SettingsConfigDict(env_prefix="MY_")
    """

    @classmethod
    def get_protocols(cls) -> tuple[type, ...]:
        """Return all protocols this class implements.

        Returns:
            Tuple of protocol types.

        """
        return ProtocolIntrospection.get_class_protocols(cls)

    def implements_protocol(self, protocol: type) -> bool:
        """Check if this instance implements a protocol.

        Args:
            protocol: The protocol type to check.

        Returns:
            True if this instance implements the protocol.

        """
        return ProtocolIntrospection.check_implements_protocol(self, protocol)


class FlextProtocolsMetaclassUtilities:
    """Static utility methods exposed by FlextProtocols facade."""

    @staticmethod
    def check_implements_protocol(
        instance: FlextProtocolsBase.Base | t.Container,
        protocol: type,
    ) -> bool:
        """Check if an instance's class implements a protocol.

        Args:
            instance: The item to check.
            protocol: The protocol to check against.

        Returns:
            True if the instance implements the protocol.

        """
        return FlextProtocolsBase.check_protocol_compliance(instance, protocol)

    @staticmethod
    def implements(*protocols: type) -> Callable[[type[T]], type[T]]:
        """Decorator to mark non-Pydantic classes as implementing protocols.

        Validates protocol compliance at class definition time and adds
        protocol introspection capabilities to the decorated class.

        This decorator is for classes that don't inherit from Pydantic
        BaseModel or BaseSettings. For Pydantic classes, use ProtocolModel
        or ProtocolSettings base classes instead.

        Usage:
            @p.implements(p.Handler, p.Domain.Repository)
            class MyHandler(FlextHandlers[Command, Result]):
                def handle(self, message: Command) -> Result:
                    ...

            # Check protocols at runtime
            handler = MyHandler()
            assert handler.implements_protocol(p.Handler)
            assert MyHandler.get_protocols() == (p.Handler, p.Domain.Repository)

        Args:
            *protocols: Protocol types that the class implements.

        Returns:
            A decorator that validates and registers protocols on the class.

        """

        def decorator(cls: type[T]) -> type[T]:
            class_name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
            for protocol in protocols:
                FlextProtocolsBase.validate_protocol_compliance(
                    cls, protocol, class_name
                )
            setattr(cls, "__protocols__", tuple(protocols))

            def _instance_implements_protocol(
                self: FlextProtocolsBase.Base | t.Container,
                protocol: type,
            ) -> bool:
                return FlextProtocolsBase.check_protocol_compliance(self, protocol)

            setattr(cls, "implements_protocol", _instance_implements_protocol)

            def _class_get_protocols(kls: type) -> tuple[type, ...]:
                return ProtocolIntrospection.get_class_protocols(kls)

            setattr(cls, "get_protocols", classmethod(_class_get_protocols))
            return cls

        return decorator

    @staticmethod
    def is_protocol(target_cls: type) -> bool:
        """Check if a class is a typing.Protocol."""
        return ProtocolIntrospection.is_protocol(target_cls)

    implements_protocol = check_implements_protocol


__all__ = [
    "FlextProtocolsMetaclassUtilities",
    "ProtocolModel",
    "ProtocolModelMeta",
    "ProtocolSettings",
    "_CombinedModelMeta",
]
