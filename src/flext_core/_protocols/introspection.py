"""Protocol introspection utilities for structural compliance checks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from flext_core import t

if TYPE_CHECKING:
    from flext_core.protocols import FlextProtocols


class _ProtocolIntrospection:
    """Internal helpers for protocol detection and compliance checks."""

    @staticmethod
    def get_protocol_attrs(protocol: type) -> tuple[str, ...]:
        raw_attrs_candidate: object = getattr(protocol, "__protocol_attrs__", ())
        if not isinstance(raw_attrs_candidate, Sequence):
            return ()
        try:
            return TypeAdapter(tuple[str, ...]).validate_python(raw_attrs_candidate)
        except ValidationError:
            return ()

    @classmethod
    def check_implements_protocol(
        cls,
        instance: FlextProtocols.Base | t.Container,
        protocol: type,
    ) -> bool:
        """Check if an instance implements a protocol."""
        registered_protocols = cls.get_class_protocols(instance.__class__)
        if protocol in registered_protocols:
            return True
        protocol_annotations: Mapping[str, type | str] = (
            protocol.__annotations__ if hasattr(protocol, "__annotations__") else {}
        )
        raw_attrs = set(cls.get_protocol_attrs(protocol))
        protocol_methods: set[str] = set()
        protocol_methods.update(raw_attrs)
        required_members: set[str] = set(protocol_annotations.keys())
        required_members.update(protocol_methods)
        required_members = {
            m
            for m in required_members
            if not m.startswith("_")
            or m.startswith("__")
            or (m in {"metadata_extra", "sealed"})
        }
        if not required_members:
            return False
        return all(hasattr(instance, member) for member in required_members)

    @classmethod
    def partition_protocol_bases(
        cls, bases: tuple[type, ...]
    ) -> tuple[list[type], list[type]]:
        """Separate Protocol bases from regular class bases."""
        protocols: list[type] = []
        model_bases: list[type] = []
        for base in bases:
            if cls.is_protocol(base):
                protocols.append(base)
            else:
                model_bases.append(base)
        return (protocols, model_bases)

    @staticmethod
    def get_class_protocols(target_cls: type) -> tuple[type, ...]:
        """Get the protocols a class implements."""
        raw_protocols: object = getattr(target_cls, "__protocols__", ())
        if not isinstance(raw_protocols, Sequence):
            return ()
        try:
            return TypeAdapter(tuple[type, ...]).validate_python(raw_protocols)
        except ValidationError:
            return ()

    @staticmethod
    def is_protocol(target_cls: type) -> bool:
        """Check if a class is a typing.Protocol."""
        is_proto = getattr(target_cls, "_is_protocol", False)
        if callable(is_proto):
            return bool(is_proto())
        return bool(is_proto)

    @staticmethod
    def validate_protocol_compliance(
        target_cls: type, protocol: type, class_name: str
    ) -> None:
        """Validate that a class implements all required protocol members."""
        protocol_annotations: Mapping[str, type | str] = (
            protocol.__annotations__ if hasattr(protocol, "__annotations__") else {}
        )
        raw_attrs = set(_ProtocolIntrospection.get_protocol_attrs(protocol))
        protocol_methods: set[str] = set()
        protocol_methods.update(raw_attrs)
        required_members: set[str] = set(protocol_annotations.keys())
        if protocol_methods:
            required_members.update(protocol_methods)
        required_members = {
            m
            for m in required_members
            if not m.startswith("_")
            or m.startswith("__")
            or (m in {"metadata_extra", "sealed"})
        }
        all_annotations: set[str] = set()
        for base in target_cls.mro():
            base_annotations: Mapping[str, type | str] = (
                base.__annotations__ if hasattr(base, "__annotations__") else {}
            )
            all_annotations.update(base_annotations.keys())
        missing = [
            member
            for member in required_members
            if not (hasattr(target_cls, member) or member in all_annotations)
        ]
        if missing:
            protocol_name = (
                protocol.__name__ if hasattr(protocol, "__name__") else str(protocol)
            )
            missing_str = ", ".join(sorted(missing))
            msg = f"Class '{class_name}' does not implement required members of protocol '{protocol_name}': {missing_str}"
            raise TypeError(msg)


_METACLASS_STRICT: bool = os.environ.get("FLEXT_METACLASS_STRICT", "1") == "1"


__all__ = ["_METACLASS_STRICT", "_ProtocolIntrospection"]
