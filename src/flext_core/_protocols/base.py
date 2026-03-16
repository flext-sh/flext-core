"""FlextProtocolsBase - foundational protocol primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, runtime_checkable

from pydantic import BaseModel

from flext_core import t
from flext_core._protocols.introspection import _ProtocolIntrospection

if TYPE_CHECKING:
    from flext_core.protocols import FlextProtocols


class FlextProtocolsBase:
    """Hierarchical protocol namespace organized by Interface Segregation Principle.

    Hierarchy follows architectural layers:
    - Base: Fundamental interfaces
    - Core: Result handling and model protocols
    - Configuration: Config and context management
    - Infrastructure: DI and container protocols
    - Domain: Business domain protocols
    - Application: CQRS and application layer protocols
    - Utility: Supporting utility protocols
    """

    @runtime_checkable
    class Base(Protocol):
        """Base protocol for FLEXT structural types."""

        pass

    @runtime_checkable
    class Model(Base, Protocol):
        """Structural typing protocol for Pydantic v2 models.

        Ensures types have Pydantic signatures without importing BaseModel directly
        in typings.py, preventing circular dependencies.
        """

        model_config: ClassVar[Mapping[str, t.Container]]
        model_fields: ClassVar[Mapping[str, type | str]]

        def model_dump(
            self, **kwargs: t.Container
        ) -> Mapping[str, t.NormalizedValue | BaseModel]:
            """Dump model to dictionary."""
            ...

        @classmethod
        def model_validate(
            cls,
            obj: t.NormalizedValue | BaseModel,
            **kwargs: t.Container,
        ) -> Self:
            """Validate object against model."""
            ...

        def validate(self) -> FlextProtocols.Result[bool]:
            """Validate model."""
            ...

    @runtime_checkable
    class Routable(Protocol):
        """Protocol for messages that carry explicit route information."""

        @property
        def command_type(self) -> str | None:
            """Command type identifier."""
            ...

        @property
        def event_type(self) -> str | None:
            """Event type identifier."""
            ...

        @property
        def query_type(self) -> str | None:
            """Query type identifier."""
            ...

    _protocol_members_cache: ClassVar[dict[type, frozenset[str]]] = {}
    _class_annotations_cache: ClassVar[dict[type, frozenset[str]]] = {}
    _compliance_results_cache: ClassVar[dict[tuple[type, type], bool]] = {}

    @classmethod
    def _get_protocol_members(cls, protocol: type) -> frozenset[str]:
        if protocol not in cls._protocol_members_cache:
            cls._protocol_members_cache[protocol] = frozenset(
                _ProtocolIntrospection.get_protocol_attrs(protocol)
            )
        return cls._protocol_members_cache[protocol]

    @classmethod
    def _get_class_annotation_members(cls, target_cls: type) -> frozenset[str]:
        if target_cls not in cls._class_annotations_cache:
            all_annotations: set[str] = set()
            for base in target_cls.mro():
                base_annotations: Mapping[str, type | str] = (
                    base.__annotations__ if hasattr(base, "__annotations__") else {}
                )
                all_annotations.update(base_annotations.keys())
            cls._class_annotations_cache[target_cls] = frozenset(all_annotations)
        return cls._class_annotations_cache[target_cls]

    @classmethod
    def _get_protocol_required_members(cls, protocol: type) -> frozenset[str]:
        protocol_annotations: Mapping[str, type | str] = (
            protocol.__annotations__ if hasattr(protocol, "__annotations__") else {}
        )
        required_members: set[str] = set(protocol_annotations.keys())
        required_members.update(cls._get_protocol_members(protocol))
        filtered_members = {
            member
            for member in required_members
            if not member.startswith("_")
            or member.startswith("__")
            or (member in {"metadata_extra", "sealed"})
        }
        return frozenset(filtered_members)

    @classmethod
    def _get_compliance_cache_key(
        cls, target_cls: type, protocol: type
    ) -> tuple[type, type]:
        return (target_cls, protocol)

    @classmethod
    def _check_protocol_compliance(
        cls,
        instance: FlextProtocols.Base | t.Container,
        protocol: type,
    ) -> bool:
        target_cls = instance.__class__
        cache_key = cls._get_compliance_cache_key(target_cls, protocol)
        cached_result = cls._compliance_results_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        try:
            runtime_compliant = isinstance(instance, protocol)
        except TypeError:
            runtime_compliant = False
        if runtime_compliant:
            cls._compliance_results_cache[cache_key] = True
            return True
        registered_protocols = _ProtocolIntrospection.get_class_protocols(target_cls)
        if protocol in registered_protocols:
            cls._compliance_results_cache[cache_key] = True
            return True
        required_members = cls._get_protocol_required_members(protocol)
        if not required_members:
            cls._compliance_results_cache[cache_key] = False
            return False
        class_annotations = cls._get_class_annotation_members(target_cls)
        is_compliant = all(
            hasattr(instance, member) or member in class_annotations
            for member in required_members
        )
        cls._compliance_results_cache[cache_key] = is_compliant
        return is_compliant

    @classmethod
    def _validate_protocol_compliance(
        cls, target_cls: type, protocol: type, class_name: str
    ) -> None:
        cache_key = cls._get_compliance_cache_key(target_cls, protocol)
        cached_result = cls._compliance_results_cache.get(cache_key)
        if cached_result is True:
            return
        required_members = cls._get_protocol_required_members(protocol)
        class_annotations = cls._get_class_annotation_members(target_cls)
        missing = [
            member
            for member in required_members
            if not (hasattr(target_cls, member) or member in class_annotations)
        ]
        if missing:
            cls._compliance_results_cache[cache_key] = False
            protocol_name = (
                protocol.__name__ if hasattr(protocol, "__name__") else str(protocol)
            )
            missing_str = ", ".join(sorted(missing))
            msg = f"Class '{class_name}' does not implement required members of protocol '{protocol_name}': {missing_str}"
            raise TypeError(msg)
        cls._compliance_results_cache[cache_key] = True


__all__ = ["FlextProtocolsBase"]
