"""Enforcement item-collection layer: project detection + per-rule iterators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from enum import EnumType
from pathlib import Path

from flext_core._constants import FlextConstantsEnforcement
from flext_core._models import FlextModelsPydantic
from flext_core._protocols import FlextProtocolsBase
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.pydantic import FlextTypesPydantic
from flext_core._utilities import (
    FlextUtilitiesBeartypeEngine,
    FlextUtilitiesProjectMetadata,
)
from flext_core._utilities.enforcement_emit import FlextUtilitiesEnforcementEmit

_ERR_ENFORCEMENT_NAMESPACE_METADATA = (
    "Cannot read project metadata for enforcement namespace resolution"
)
_ERR_ENFORCEMENT_CLASS_STEM_METADATA = (
    "Cannot read project metadata for enforcement class stem override"
)
_ERR_ENFORCEMENT_NAMESPACE_SOURCE = (
    "Cannot inspect target source module for namespace resolution"
)


class FlextUtilitiesEnforcementCollect(FlextUtilitiesEnforcementEmit):
    """Project resolution + rule-input iterators."""

    @staticmethod
    def _owning_project_root(target: type) -> Path | None:
        """Return the pyproject root that physically owns the target source.

        Returns:
            The pyproject root that physically owns the target source.

        """
        source_file = FlextUtilitiesEnforcementCollect._resolve_target_source_file(
            target,
        )
        if source_file is None:
            return None
        source = source_file.resolve()
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
                top,
            ):
                # mro-j47u (codex): never attribute .venv/site-packages classes
                # to the consuming project's pyproject and namespace prefix.
                return parent
        return None

    @staticmethod
    def _resolve_target_source_file(target: type) -> Path | None:
        """Resolve target source file path with explicit error semantics.

        Returns:
            The resulting ``Path | None``.

        Raises:
            RuntimeError: If a ``(OSError, TypeError)`` is caught.

        """
        try:
            src_file = inspect.getsourcefile(target)
        except (OSError, TypeError) as exc:
            raise RuntimeError(_ERR_ENFORCEMENT_NAMESPACE_SOURCE) from exc
        if src_file is None:
            return None
        return Path(src_file)

    @staticmethod
    def _discover_src_package(target: type) -> str | None:
        """Return the package owned by the target's physical project root.

        Source-skip on built-in classes / dynamic types per the runtime-safety
        contract: ``inspect.getsourcefile`` raises ``TypeError`` for built-ins
        and ``OSError`` when the source file cannot be read; both produce
        ``None`` here so the dispatcher cleanly skips the target.

        Returns:
            The package owned by the target's physical project root.

        Raises:
            RuntimeError: If a ``(OSError, ValueError)`` is caught.

        """
        try:
            project_root = FlextUtilitiesEnforcementCollect._owning_project_root(target)
        except RuntimeError:
            project_root = None
        if project_root is None:
            return None
        top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
        if (project_root / "src" / top).is_dir() or (project_root / top).is_dir():
            return top
        try:
            document = FlextUtilitiesProjectMetadata.read_project_document_cached(
                project_root,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(_ERR_ENFORCEMENT_NAMESPACE_METADATA) from exc
        if document.project is None:
            return None
        return FlextUtilitiesProjectMetadata.build_project_metadata(
            project_root,
            document,
        ).package_name

    @staticmethod
    def _project(target: type) -> FlextTypingBase.StrPair | None:
        """Return (derived_prefix, inner_namespace) or None if unknowable.

        Returns:
            (derived_prefix, inner_namespace) or None if unknowable.

        Raises:
            RuntimeError: If a ``(OSError, ValueError)`` is caught.

        """
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
                try:
                    document = (
                        FlextUtilitiesProjectMetadata.read_project_document_cached(
                            project_root,
                        )
                    )
                except (OSError, ValueError) as exc:
                    raise RuntimeError(_ERR_ENFORCEMENT_CLASS_STEM_METADATA) from exc
                metadata = FlextUtilitiesProjectMetadata.build_project_metadata(
                    project_root,
                    document,
                )
                class_stem_override = metadata.flext.project.class_stem_override
        canonical_project_name = src.replace("_", "-")
        head, _, tail = canonical_project_name.partition("-")
        namespace = FlextUtilitiesProjectMetadata.derive_class_stem(tail or head)
        project_prefix = (
            class_stem_override
            or FlextUtilitiesProjectMetadata.derive_class_stem(
                canonical_project_name,
            )
        )
        if top in {"tests", "examples", "scripts"} and top != (src or ""):
            return FlextUtilitiesProjectMetadata.derive_class_stem(
                top,
            ) + project_prefix, namespace
        return project_prefix, namespace

    @staticmethod
    def _iter_inner(target: type) -> Iterator[tuple[str, type]]:
        """Yield the public classes declared inside ``target``.

        A class the body only references (``REGEX_ERROR = re.error``) is not
        one of the target's declarations and is never audited as the project's.

        Yields:
            Each ``tuple[str, type]``.

        """
        for name, value in vars(target).items():
            if (
                isinstance(value, type)
                and not name.startswith("_")
                and FlextUtilitiesBeartypeEngine.defined_inside(
                    value,
                    target.__qualname__,
                )
            ):
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
        # ``model_fields`` resolves through ``BaseModel``; settings classes
        # subclass it, so the bare model base is the canonical collector type.
        model_type: type[FlextModelsPydantic.BaseModel],
        tag: str,
    ) -> Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]]:
        own_ann = set(vars(model_type).get("__annotations__", {}))
        for name, info in model_type.model_fields.items():
            if name not in own_ann:
                continue
            args: tuple[FlextProtocolsBase.AttributeProbe, ...] = (
                (model_type, name, info)
                if tag in {"missing_description", "no_inline_union"}
                else (info,)
            )
            yield f'Field "{name}"', args

    @staticmethod
    def _attr_filter(
        target: type,
        layer: str,
    ) -> Callable[[str, FlextTypesPydantic.JsonValue], bool]:
        if layer == FlextConstantsEnforcement.EnforcementLayer.CONSTANTS.lower():
            accept: Callable[[str, FlextTypesPydantic.JsonValue], bool] = (
                FlextUtilitiesBeartypeEngine.attr_accept_constants
            )
            return accept
        if layer == FlextConstantsEnforcement.EnforcementLayer.UTILITIES.lower():

            def accept_utility(name: str, _value: FlextTypesPydantic.JsonValue) -> bool:
                allowed: bool = FlextUtilitiesBeartypeEngine.attr_accept_utility(
                    target,
                    name,
                )
                return allowed

            return accept_utility

        def accept_public(name: str, _value: FlextTypesPydantic.JsonValue) -> bool:
            allowed: bool = FlextUtilitiesBeartypeEngine.attr_accept_public(name)
            return allowed

        return accept_public

    @staticmethod
    def _attr_items(
        target: type,
        layer: str,
    ) -> Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]]:
        accept = FlextUtilitiesEnforcementCollect._attr_filter(target, layer)
        qn = target.__qualname__
        for name, value in vars(target).items():
            if accept(name, value):
                yield f"{qn}.{name}", (name, value)

    @staticmethod
    def _ns_class_prefix(
        target: type,
        qn: str,
        project: FlextTypingBase.StrPair,
    ) -> Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]]:
        skip_roots = (
            FlextConstantsEnforcement.ENFORCEMENT_NAMESPACE_FACADE_ROOTS
            | FlextConstantsEnforcement.ENFORCEMENT_INFRASTRUCTURE_BASES
        )
        if "." in qn or target.__name__ in skip_roots:
            return
        yield qn, (target, project[0])

    @staticmethod
    def _ns_cross(
        target: type,
        qn: str,
        effective_layer: str,
    ) -> Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]]:
        layer = (
            effective_layer
            or FlextUtilitiesEnforcementCollect.detect_layer(target)
            or ""
        )

        def walk(
            node: type,
            path: str,
        ) -> Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]]:
            for name, value in FlextUtilitiesEnforcementCollect._iter_inner(node):
                full = f"{path}.{name}"
                yield full, (value, layer)
                if not isinstance(value, EnumType):
                    yield from walk(value, full)

        yield from walk(target, qn)


__all__: list[str] = ["FlextUtilitiesEnforcementCollect"]
