"""Enforcement item-collection layer: project detection + per-rule iterators."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from enum import EnumType
from pathlib import Path

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as ub
from flext_core._utilities.enforcement_emit import FlextUtilitiesEnforcementEmit
from flext_core._utilities.project_metadata import FlextUtilitiesProjectMetadata as upm


class FlextUtilitiesEnforcementCollect(FlextUtilitiesEnforcementEmit):
    """Project resolution + rule-input iterators."""

    @staticmethod
    def _discover_src_package(target: type) -> str | None:
        """Walk parents to the first ``pyproject.toml`` and return the package name."""
        src_file = inspect.getsourcefile(target)
        if src_file is None:
            return None
        for parent in Path(src_file).parents:
            pyproject = parent / "pyproject.toml"
            if not pyproject.exists():
                continue
            src = parent / "src"
            if src.is_dir():
                for child in src.iterdir():
                    if child.is_dir() and (child / "__init__.py").exists():
                        return child.name
            meta = upm.read_project_metadata(parent)
            return meta.package_name
        return None

    @staticmethod
    def _project(target: type) -> tuple[str, str] | None:
        """Return (derived_prefix, inner_namespace) or None if unknowable."""
        top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
        if not top:
            return None
        src = FlextUtilitiesEnforcementCollect._discover_src_package(target)
        if src is None:
            if top == "fence":
                return None
            src = top
        head, _, tail = src.partition("_")
        namespace = upm.pascalize(tail or head)
        project_prefix = cpm.SPECIAL_NAME_OVERRIDES.get(
            src.replace("_", "-"),
        ) or upm.pascalize(src)
        if top in {"tests", "examples", "scripts"} and top != (src or ""):
            return upm.pascalize(top) + project_prefix, namespace
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
        model_type: type[mp.BaseModel],
        tag: str,
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
        own_ann = set(vars(model_type).get("__annotations__", {}))
        for name, info in model_type.model_fields.items():
            if name not in own_ann:
                continue
            args: tuple[object, ...] = (
                (model_type, name, info) if tag == "missing_description" else (info,)
            )
            yield f'Field "{name}"', args

    @staticmethod
    def _attr_filter(layer: str) -> Callable[[str, tp.JsonValue], bool]:
        if layer == c.EnforcementLayer.CONSTANTS.lower():
            return ub.attr_accept_constants
        if layer == c.EnforcementLayer.UTILITIES.lower():
            return lambda name, _v: ub.attr_accept_utility(name)
        return lambda name, _v: ub.attr_accept_public(name)

    @staticmethod
    def _attr_items(
        target: type,
        layer: str,
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
        accept = FlextUtilitiesEnforcementCollect._attr_filter(layer)
        qn = target.__qualname__
        for name, value in vars(target).items():
            if accept(name, value):
                yield f"{qn}.{name}", (name, value)

    @staticmethod
    def _ns_class_prefix(
        target: type,
        qn: str,
        project: tuple[str, str],
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
        skip_roots = (
            c.ENFORCEMENT_NAMESPACE_FACADE_ROOTS | c.ENFORCEMENT_INFRASTRUCTURE_BASES
        )
        if "." in qn or target.__name__ in skip_roots:
            return
        yield qn, (target, project[0])

    @staticmethod
    def _ns_cross(
        target: type,
        qn: str,
        effective_layer: str,
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
        layer = (
            effective_layer
            or FlextUtilitiesEnforcementCollect.detect_layer(target)
            or ""
        )

        def walk(node: type, path: str) -> Iterator[tuple[str, tuple[object, ...]]]:
            for name, value in FlextUtilitiesEnforcementCollect._iter_inner(node):
                if not ub.defined_inside(value, node.__qualname__):
                    continue
                full = f"{path}.{name}"
                yield full, (value, layer)
                if not isinstance(value, EnumType):
                    yield from walk(value, full)

        yield from walk(target, qn)

    @staticmethod
    def _ns_nested_mro(
        target: type,
        qn: str,
        project: tuple[str, str],
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
        top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
        if top and top == FlextUtilitiesEnforcementCollect._discover_src_package(
            target
        ):
            return
        yield qn, (target, project[0])

    @staticmethod
    def _ns_no_accessor_methods(
        target: type,
        qn: str,
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
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
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
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
        match tag:
            case "class_prefix":
                yield from cls._ns_class_prefix(target, qn, project)
            case "cross_strenum" | "cross_protocol":
                yield from cls._ns_cross(target, qn, effective_layer)
            case "nested_mro":
                yield from cls._ns_nested_mro(target, qn, project)
            case "no_accessor_methods":
                yield from cls._ns_no_accessor_methods(target, qn)
            case "settings_inheritance":
                yield qn, (target,)
            # --- Lane A-PT dispatch arms (ENFORCE-039/041/043/044) ---
            case (
                "cast_outside_core"
                | "model_rebuild_call"
                | "pass_through_wrapper"
                | "private_attr_probe"
            ):
                yield qn, (target,)
            case _:
                return


__all__: list[str] = ["FlextUtilitiesEnforcementCollect"]
