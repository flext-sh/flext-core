"""Enforcement item-collection layer: project detection + per-rule iterators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from enum import EnumType

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._protocols.base import FlextProtocolsBase as pb
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._enforcement_collect_parts.enforcement_collect_part_01 import (
    FlextUtilitiesEnforcementCollect as FlextUtilitiesEnforcementCollectPart01,
)
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as ub


class FlextUtilitiesEnforcementCollect(FlextUtilitiesEnforcementCollectPart01):
    @staticmethod
    def _ns_nested_mro(
        target: type,
        qn: str,
        project: t.StrPair,
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
        if top and top == FlextUtilitiesEnforcementCollect._discover_src_package(
            target,
        ):
            return
        yield qn, (target, project[0])

    @staticmethod
    def _ns_no_accessor_methods(
        target: type,
        qn: str,
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        for name, value in vars(target).items():
            if inspect.isfunction(value) or isinstance(
                value,
                (classmethod, staticmethod),
            ):
                yield f"{qn}.{name}", (target, name)

    @staticmethod
    def _ns_classvar_constants(
        target: type,
        qn: str,
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        """Yield one item per public attribute; the visitor judges each one.

        Granularity lives here (the iterator), never in the visitor: every
        violating constant of the class surfaces in one pass instead of one
        per gate round.

        Yields:
            Each ``tuple[str, tuple[pb.AttributeProbe, ...]]``.

        """
        for name, value in vars(target).items():
            if name.startswith("_") or name != name.upper():
                continue
            yield f"{qn}.{name}", (target, name, value)

    @staticmethod
    def _ns_nested_classes(
        root: type,
        node: type,
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        """Yield every locally declared non-Enum class nested under ``node``.

        Yields:
            Each ``tuple[str, tuple[pb.AttributeProbe, ...]]``.

        """
        for value in vars(node).values():
            if (
                isinstance(value, type)
                and not isinstance(value, EnumType)
                and ub.defined_inside(value, node.__qualname__)
            ):
                yield value.__qualname__, (root, value)
                yield from FlextUtilitiesEnforcementCollect._ns_nested_classes(
                    root,
                    value,
                )

    @staticmethod
    def _namespace_items(
        target: type,
        tag: str,
        effective_layer: str = "",
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        """Per-tag dispatcher for namespace-category rule inputs.

        Yields:
            Each ``tuple[str, tuple[pb.AttributeProbe, ...]]``.

        Raises:
            ValueError: If unknown namespace collection.

        """
        if (
            ub.defined_in_function_scope(target)
            or target.__name__.startswith("_")
            or "[" in target.__name__
        ):
            return
        qn = target.__qualname__
        project = FlextUtilitiesEnforcementCollect._project(target)
        if project is None:
            return
        cls = FlextUtilitiesEnforcementCollect
        # The collection strategy is rule data; an unknown strategy is a data
        # defect and raises instead of collecting nothing.
        collect = c.ENFORCEMENT_TAG_COLLECT[tag]
        match collect:
            case "target":
                yield qn, (target,)
            case "class_prefix":
                yield from cls._ns_class_prefix(target, qn, project)
            case "cross_layer":
                yield from cls._ns_cross(target, qn, effective_layer)
            case "nested_classes":
                yield from cls._ns_nested_classes(target, target)
            case "nested_mro":
                yield from cls._ns_nested_mro(target, qn, project)
            case "accessor_methods":
                yield from cls._ns_no_accessor_methods(target, qn)
            case "classvar_constants":
                yield from cls._ns_classvar_constants(target, qn)
            case _:
                msg = f"unknown namespace collection {collect!r} for tag {tag!r}"
                raise ValueError(msg)


__all__: list[str] = ["FlextUtilitiesEnforcementCollect"]
