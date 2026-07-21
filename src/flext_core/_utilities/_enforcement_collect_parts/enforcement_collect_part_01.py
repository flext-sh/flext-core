"""Enforcement item-collection layer: project detection + per-rule iterators."""

from __future__ import annotations

import inspect

# mro-cqxy (ADR-011): runtime import for runtime-evaluated annotation
from collections.abc import Callable
from enum import EnumType
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._protocols.base import FlextProtocolsBase as pb
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as ub
from flext_core._utilities.enforcement_emit import FlextUtilitiesEnforcementEmit
from flext_core._utilities.project_metadata import FlextUtilitiesProjectMetadata as upm

if TYPE_CHECKING:
    from collections.abc import Iterator

    from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesEnforcementCollect(FlextUtilitiesEnforcementEmit):
    """Project resolution + rule-input iterators."""

    @staticmethod
    def _owning_project_root(target: type) -> Path | None:
        """Return the pyproject root that physically owns the target source."""
        try:
            src_file = inspect.getsourcefile(target)
        except (OSError, TypeError):
            return None
        if src_file is None:
            return None
        source = Path(src_file).resolve()
        top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
        if not top:
            return None
        for parent in source.parents:
            if not (parent / "pyproject.toml").is_file():
                continue
            try:
                relative = source.relative_to(parent)
            except ValueError:
                continue
            if relative.is_relative_to(Path("src") / top) or relative.is_relative_to(
                top
            ):
                # mro-j47u (codex): never attribute .venv/site-packages classes
                # to the consuming project's pyproject and namespace prefix.
                return parent
        return None

    @staticmethod
    def _discover_src_package(target: type) -> str | None:
        """Return the package owned by the target's physical project root.

        Source-skip on built-in classes / dynamic types per the runtime-safety
        contract: ``inspect.getsourcefile`` raises ``TypeError`` for built-ins
        and ``OSError`` when the source file cannot be read; both produce
        ``None`` here so the dispatcher cleanly skips the target.
        """
        project_root = FlextUtilitiesEnforcementCollect._owning_project_root(target)
        if project_root is None:
            return None
        top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
        if (project_root / "src" / top).is_dir() or (project_root / top).is_dir():
            return top
        metadata_result = upm.read_project_metadata(project_root)
        if metadata_result.failure:
            return None
        package_name: str = metadata_result.value.package_name
        return package_name

    @staticmethod
    def _project(target: type) -> t.StrPair | None:
        """Return (derived_prefix, inner_namespace) or None if unknowable."""
        top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
        if not top:
            return None
        src = FlextUtilitiesEnforcementCollect._discover_src_package(target)
        if src is None:
            if top == "fence":
                return None
            src = top
            class_stem_override = None
        else:
            class_stem_override = None
            project_root = FlextUtilitiesEnforcementCollect._owning_project_root(target)
            if project_root is not None:
                metadata_result = upm.read_project_metadata(project_root)
                if metadata_result.failure:
                    return None
                class_stem_override = (
                    metadata_result.value.flext.project.class_stem_override
                )
        canonical_project_name = src.replace("_", "-")
        head, _, tail = canonical_project_name.partition("-")
        namespace = upm.derive_class_stem(tail or head)
        project_prefix = class_stem_override or upm.derive_class_stem(
            canonical_project_name
        )
        if top in {"tests", "examples", "scripts"} and top != (src or ""):
            return upm.derive_class_stem(top) + project_prefix, namespace
        return project_prefix, namespace

    @staticmethod
    def _iter_inner(target: type) -> Iterator[tuple[str, type]]:
        for name, value in vars(target).items():
            if isinstance(value, type) and not name.startswith("_"):
                yield name, value

    @staticmethod
    def _iter_effective(target: type) -> Iterator[tuple[str, type]]:
        direct = list(FlextUtilitiesEnforcementCollect._iter_inner(target))
        if direct:
            yield from direct
            return
        seen: set[str] = set()
        for base in target.__mro__[1:]:
            if base is object:
                continue
            for name, value in FlextUtilitiesEnforcementCollect._iter_inner(base):
                if name not in seen:
                    seen.add(name)
                    yield name, value

    @staticmethod
    def _field_items(
        model_type: type[tp.BaseModelType], tag: str
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        own_ann = set(vars(model_type).get("__annotations__", {}))
        for name, info in model_type.__pydantic_fields__.items():
            if name not in own_ann:
                continue
            args: tuple[pb.AttributeProbe, ...] = (
                (model_type, name, info) if tag == "missing_description" else (info,)
            )
            yield f'Field "{name}"', args

    @staticmethod
    def _attr_filter(layer: str) -> Callable[[str, tp.JsonValue], bool]:
        if layer == c.EnforcementLayer.CONSTANTS.lower():
            accept: Callable[[str, tp.JsonValue], bool] = ub.attr_accept_constants
            return accept
        if layer == c.EnforcementLayer.UTILITIES.lower():

            def accept_utility(name: str, _value: tp.JsonValue) -> bool:
                allowed: bool = ub.attr_accept_utility(name)
                return allowed

            return accept_utility

        def accept_public(name: str, _value: tp.JsonValue) -> bool:
            allowed: bool = ub.attr_accept_public(name)
            return allowed

        return accept_public

    @staticmethod
    def _attr_items(
        target: type, layer: str
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        accept = FlextUtilitiesEnforcementCollect._attr_filter(layer)
        qn = target.__qualname__
        for name, value in vars(target).items():
            if accept(name, value):
                yield f"{qn}.{name}", (name, value)

    @staticmethod
    def _ns_class_prefix(
        target: type, qn: str, project: t.StrPair
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        skip_roots = (
            c.ENFORCEMENT_NAMESPACE_FACADE_ROOTS | c.ENFORCEMENT_INFRASTRUCTURE_BASES
        )
        if "." in qn or target.__name__ in skip_roots:
            return
        yield qn, (target, project[0])

    @staticmethod
    def _ns_cross(
        target: type, qn: str, effective_layer: str
    ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
        layer = (
            effective_layer
            or FlextUtilitiesEnforcementCollect.detect_layer(target)
            or ""
        )

        def walk(
            node: type, path: str
        ) -> Iterator[tuple[str, tuple[pb.AttributeProbe, ...]]]:
            for name, value in FlextUtilitiesEnforcementCollect._iter_inner(node):
                if not ub.defined_inside(value, node.__qualname__):
                    continue
                full = f"{path}.{name}"
                yield full, (value, layer)
                if not isinstance(value, EnumType):
                    yield from walk(value, full)

        yield from walk(target, qn)


__all__: list[str] = ["FlextUtilitiesEnforcementCollect"]
