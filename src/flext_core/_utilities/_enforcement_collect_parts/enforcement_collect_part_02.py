"""Enforcement item-collection layer: project detection + per-rule iterators."""

from __future__ import annotations

import inspect
from collections.abc import Iterator

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._protocols.base import FlextProtocolsBase as pb
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as ub

from .enforcement_collect_part_01 import (
    FlextUtilitiesEnforcementCollect as FlextUtilitiesEnforcementCollectPart01,
)


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
    def _namespace_items(
        target: type,
        tag: str,
        effective_layer: str = "",
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        """Per-tag dispatcher for namespace-category rule inputs."""
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
        if tag in c.ENFORCEMENT_NAMESPACE_TARGET_TAGS:
            yield qn, (target,)
            return
        match tag:
            case "class_prefix":
                yield from cls._ns_class_prefix(target, qn, project)
            case "cross_strenum" | "cross_protocol":
                yield from cls._ns_cross(target, qn, effective_layer)
            case "nested_mro":
                yield from cls._ns_nested_mro(target, qn, project)
            case "no_accessor_methods":
                yield from cls._ns_no_accessor_methods(target, qn)
            case _:
                return


__all__: list[str] = ["FlextUtilitiesEnforcementCollect"]
