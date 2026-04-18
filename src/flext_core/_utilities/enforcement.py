"""Runtime enforcement engine — rules are DATA in c.*, predicates are METHODS in ub.*.

Every row in ``c.ENFORCEMENT_RULES`` is paired by naming convention with a
``check_<tag>`` staticmethod on ``FlextUtilitiesBeartypeEngine``. This module
holds only the dispatch engine + violation assembly; no rule strings, no
predicate bodies, no back-compat shims.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Iterator, Mapping
from enum import EnumType
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic.fields import FieldInfo

from flext_core import (
    FlextConstantsEnforcement as c,
    FlextModelsEnforcement as me,
    FlextModelsPydantic as mp,
    FlextUtilitiesBeartypeEngine as ub,
)
from flext_core._constants.project_metadata import (
    FlextConstantsProjectMetadata as _kpm,
)
from flext_core._utilities.project_metadata import (
    FlextUtilitiesProjectMetadata as _ump,
)

if TYPE_CHECKING:
    from flext_core import t


class FlextUtilitiesEnforcement:
    """Rule-driven runtime enforcement (static-only)."""

    # ------------------------------------------------------------------
    # Violation assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _violation(
        tag: str,
        location: str,
        qualname: str,
        detail: Mapping[str, str] | None = None,
    ) -> me.Violation:
        _cat, layer, severity, problem, fix = c.ENFORCEMENT_RULES[tag]
        subs = detail or {}
        return me.Violation(
            qualname=qualname,
            layer=str(layer),
            severity=str(severity),
            message=c.ENFORCEMENT_MSG_VIOLATION.format(
                location=location,
                problem=problem.format(**subs) if subs else problem,
                fix=fix.format(**subs) if subs else fix,
            ),
        )

    @staticmethod
    def merge_reports(*reports: me.Report) -> me.Report:
        return me.Report(
            violations=[v for r in reports for v in r.violations],
        )

    @staticmethod
    def emit(report: me.Report, *, mode: c.EnforcementMode | None = None) -> None:
        if report.empty:
            return
        active = mode or c.ENFORCEMENT_MODE
        if active is c.EnforcementMode.OFF:
            return
        for v in report.violations:
            msg = (
                f"\n{v.qualname} violates FLEXT {v.layer} {v.severity}:\n  - "
                f"{v.message}\n\nFix: See AGENTS.md § {v.layer} governance."
            )
            warnings.warn(msg, UserWarning, stacklevel=4)
            if active is c.EnforcementMode.STRICT:
                raise TypeError(msg)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def detect_layer(target: type) -> str | None:
        """Infer the facade layer from the class name.

        Matches the layer keyword (``Constants`` / ``Models`` / ``Protocols``
        / ``Types`` / ``Utilities``) anywhere in the class name — not only
        at the end — so composed facades such as ``FooConstantsSettings``
        or ``FooProtocolsBase`` still get the correct layer routing.
        Generic-specialization brackets (``Foo[Bar]``) are stripped first
        so the search ignores type-parameter noise.
        """
        name = target.__name__.partition("[")[0]
        for suffix, layer in c.ENFORCEMENT_NAMESPACE_LAYER_MAP:
            if suffix in name:
                return layer
        return None

    @staticmethod
    def _discover_src_package(target: type) -> str | None:
        """Walk parents to the first ``pyproject.toml`` and return the package name.

        Prefers the ``src/<pkg>/`` layout. Falls back to the ``[project].name``
        field in ``pyproject.toml`` when no ``src/`` tree exists (workspace
        roots like ``flext/`` that host test dirs at the top level).
        """
        try:
            src_file = inspect.getfile(target)
        except (TypeError, OSError):
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
            meta = _ump.read_project_metadata(parent)
            return meta.package_name
        return None

    @staticmethod
    def _project(target: type) -> tuple[str, str] | None:
        """Return (derived_prefix, inner_namespace) or None if unknowable.

        * Explicit overrides in ``c.SPECIAL_NAME_OVERRIDES`` (SSOT)
          win over PascalCase derivation (``flext-core`` → ``Flext``;
          ``flext`` → ``FlextRoot``).
        * When a class lives under a top-level package ``tests`` /
          ``examples`` / ``scripts``, its facade prefix is that package's
          PascalCase form prepended to the project prefix — e.g. a class in
          ``tests.*`` belonging to the ``flext_core`` project must start
          with ``TestsFlext`` (derived from ``Tests`` + ``Flext``).
        """
        top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
        if not top:
            return None
        src = FlextUtilitiesEnforcement._discover_src_package(target)

        def resolve_project_prefix() -> tuple[str, str]:
            key = src if src is not None else top
            override = _kpm.SPECIAL_NAME_OVERRIDES.get(key.replace("_", "-"))
            if override is not None:
                head, _, tail = key.partition("_")
                return override, _ump.pascalize(tail or head)
            if src is None:
                return _ump.pascalize(top), _ump.pascalize(top)
            if top == src:
                head, _, tail = src.partition("_")
                return _ump.pascalize(src), _ump.pascalize(tail or head)
            # Auxiliary root (tests/examples/scripts) — return raw project
            # prefix only; the outer code prepends the auxiliary prefix.
            head, _, tail = src.partition("_")
            return _ump.pascalize(src), _ump.pascalize(tail or head)

        project_prefix, namespace = resolve_project_prefix()
        # Workspace-level auxiliary roots (tests / examples / scripts) live
        # under a top-level module of that name and wear a prefix composed
        # of that module's PascalCase plus the project prefix.
        if top in {"tests", "examples", "scripts"} and top != (src or ""):
            return _ump.pascalize(top) + project_prefix, namespace
        return project_prefix, namespace

    # ------------------------------------------------------------------
    # Field & attribute collection
    # ------------------------------------------------------------------

    @staticmethod
    def own_fields(
        model_type: type[mp.BaseModel],
    ) -> Mapping[str, FieldInfo]:
        own_ann = set(vars(model_type).get("__annotations__", {}))
        return {
            name: info
            for name, info in model_type.model_fields.items()
            if name in own_ann
        }

    @staticmethod
    def _iter_inner(target: type) -> Iterator[tuple[str, type]]:
        for name, value in vars(target).items():
            if isinstance(value, type) and not name.startswith("_"):
                yield name, value

    @staticmethod
    def _iter_effective(target: type) -> Iterator[tuple[str, type]]:
        direct = list(FlextUtilitiesEnforcement._iter_inner(target))
        if direct:
            yield from direct
            return
        seen: set[str] = set()
        for base in target.__mro__[1:]:
            if base is object:
                continue
            for name, value in FlextUtilitiesEnforcement._iter_inner(base):
                if name not in seen:
                    seen.add(name)
                    yield name, value

    # ------------------------------------------------------------------
    # Unified rule-application engine — all categories dispatch through here
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rule(
        tag: str,
        qualname: str,
        items: Iterator[tuple[str, tuple[object, ...]]],
    ) -> list[me.Violation]:
        """Call ``ub.check_<tag>(*args)`` per item; emit violation on non-None."""
        predicate: Callable[..., Mapping[str, str] | None] = getattr(
            ub,
            f"check_{tag}",
        )
        return [
            FlextUtilitiesEnforcement._violation(tag, location, qualname, detail)
            for location, args in items
            if (detail := predicate(*args)) is not None
        ]

    # ------------------------------------------------------------------
    # Category-specific item iterators (feed _apply_rule)
    # ------------------------------------------------------------------

    @staticmethod
    def _field_items(
        model_type: type[mp.BaseModel],
        tag: str,
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
        for name, info in FlextUtilitiesEnforcement.own_fields(model_type).items():
            args: tuple[object, ...] = (
                (model_type, name, info) if tag == "missing_description" else (info,)
            )
            yield f'Field "{name}"', args

    @staticmethod
    def _attr_filter(
        layer: str,
    ) -> Callable[[str, t.RecursiveContainer], bool]:
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
        accept = FlextUtilitiesEnforcement._attr_filter(layer)
        qn = target.__qualname__
        for name, value in vars(target).items():
            if accept(name, value):
                yield f"{qn}.{name}", (name, value)

    @staticmethod
    def _namespace_items(
        target: type,
        tag: str,
        effective_layer: str = "",
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
        """Yield ``(location, args)`` pairs per namespace rule tag.

        ``effective_layer`` is the layer inherited from the enclosing
        ``check()`` call — when ``check()`` recurses into inner classes
        (for recursive attr rules), the outer layer must be preserved so
        cross-layer checks don't reset to "" on the inner target.
        """
        # Dynamic context-based skips — no hardcoded name lists.
        # Pydantic/Generic specializations carry a bracketed ``__name__``
        # (``Foo[int]``) — synthetic runtime artifacts, never user facades.
        if (
            ub.is_function_local(target)
            or target.__name__.startswith("_")
            or "[" in target.__name__
        ):
            return
        qn = target.__qualname__
        project = FlextUtilitiesEnforcement._project(target)
        if project is None:
            return

        if tag == "class_prefix":
            # Top-level only — inner classes are namespaces of their outer.
            if "." in qn:
                return
            if (
                target.__name__ in c.ENFORCEMENT_NAMESPACE_FACADE_ROOTS
                or target.__name__ in c.ENFORCEMENT_INFRASTRUCTURE_BASES
            ):
                return
            yield qn, (target, project[0])
            return

        if tag in {"cross_strenum", "cross_protocol"}:
            # Prefer the layer inherited from the enclosing check() — only
            # fall back to the target's own detected layer when no context.
            layer = (
                effective_layer or FlextUtilitiesEnforcement.detect_layer(target) or ""
            )

            def walk(node: type, path: str) -> Iterator[tuple[str, tuple[object, ...]]]:
                for name, value in FlextUtilitiesEnforcement._iter_inner(node):
                    # Skip re-assigned aliases (``Status = c.Status``) — they
                    # belong to their defining module, not this walk target.
                    if not ub.is_defined_inside(value, node.__qualname__):
                        continue
                    full = f"{path}.{name}"
                    yield full, (value, layer)
                    if not isinstance(value, EnumType):
                        yield from walk(value, full)

            yield from walk(target, qn)
            return

        if tag == "nested_mro":
            top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
            if top and top == FlextUtilitiesEnforcement._discover_src_package(target):
                return
            yield qn, (target, project[0])
            return

        if tag == "no_accessor_methods":
            # Scan own-class methods only — inherited methods come from
            # Pydantic/stdlib bases and belong to those libraries.
            for name, value in vars(target).items():
                if inspect.isfunction(value) or isinstance(
                    value,
                    (classmethod, staticmethod),
                ):
                    yield f"{qn}.{name}", (target, name)
            return

        if tag == "settings_inheritance":
            # One row per target — predicate inspects the MRO.
            yield qn, (target,)

    # ------------------------------------------------------------------
    # Public query + emission API
    # ------------------------------------------------------------------

    @staticmethod
    def _items_for(
        target: type,
        tag: str,
        category: c.EnforcementCategory,
        effective_layer: str,
        rule_layer: str,
        *,
        is_model: bool,
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
        """Return category-specific (location, args) pairs for one rule tag.

        This is the single category→iterator dispatch — ``check()`` runs
        every row in ``c.ENFORCEMENT_RULES`` through here and pipes the
        result into :meth:`_apply_rule`.
        """
        # Generic/Pydantic specializations (``Foo[Bar]``) are synthetic
        # runtime artifacts shared across all categories — skip universally.
        if "[" in target.__name__:
            return
        if category is c.EnforcementCategory.FIELD:
            if is_model and issubclass(target, mp.BaseModel):
                yield from FlextUtilitiesEnforcement._field_items(
                    target,
                    tag,
                )
            return
        if category is c.EnforcementCategory.MODEL_CLASS:
            if is_model:
                yield target.__qualname__, (target,)
            return
        if category is c.EnforcementCategory.ATTR:
            if rule_layer.lower() == effective_layer:
                yield from FlextUtilitiesEnforcement._attr_items(
                    target,
                    effective_layer,
                )
            return
        if category is c.EnforcementCategory.NAMESPACE:
            yield from FlextUtilitiesEnforcement._namespace_items(
                target,
                tag,
                effective_layer,
            )
            return
        if category is c.EnforcementCategory.PROTOCOL_TREE:
            if effective_layer != c.EnforcementLayer.PROTOCOLS.lower():
                return
            iterator = (
                FlextUtilitiesEnforcement._iter_effective
                if tag == "proto_inner_kind"
                else FlextUtilitiesEnforcement._iter_inner
            )

            def walk(node: type, path: str) -> Iterator[tuple[str, tuple[object, ...]]]:
                for name, value in iterator(node):
                    nested = f"{path}.{name}"
                    yield nested, (value,)
                    if ub.is_runtime_protocol_target(value) or ub.has_nested_namespace(
                        value
                    ):
                        yield from walk(value, nested)

            yield from walk(target, target.__qualname__)

    @staticmethod
    def check(target: type, *, layer: str | None = None) -> me.Report:
        """Query all applicable rules and return a typed report (no emission).

        Every rule dispatches through the unified :meth:`_apply_rule` —
        no per-category engine duplication; item iterators live in the
        ``_*_items`` / :meth:`_items_for` helpers and vary only by tag.
        Attr-rule recursion is handled via ``c.ENFORCEMENT_RECURSIVE_TAGS``.
        """
        violations: list[me.Violation] = []
        effective_layer = layer or FlextUtilitiesEnforcement.detect_layer(target) or ""
        is_model = isinstance(target, type) and issubclass(target, mp.BaseModel)
        qn = target.__qualname__
        for tag, row in c.ENFORCEMENT_RULES.items():
            category, rule_layer, *_ = row
            items = FlextUtilitiesEnforcement._items_for(
                target,
                tag,
                category,
                effective_layer,
                str(rule_layer),
                is_model=is_model,
            )
            violations.extend(
                FlextUtilitiesEnforcement._apply_rule(
                    tag,
                    qn,
                    items,
                )
            )
            if (
                category is c.EnforcementCategory.ATTR
                and tag in c.ENFORCEMENT_RECURSIVE_TAGS
                and str(rule_layer).lower() == effective_layer
            ):
                for _name, inner in FlextUtilitiesEnforcement._iter_inner(target):
                    if isinstance(inner, EnumType):
                        continue
                    violations.extend(
                        FlextUtilitiesEnforcement.check(
                            inner,
                            layer=effective_layer,
                        ).violations
                    )
        return me.Report(violations=violations)

    @staticmethod
    def run(model_type: type[mp.BaseModel]) -> None:
        """Pydantic ``__pydantic_init_subclass__`` hook.

        Function-local classes (Python's ``<locals>`` qualname marker)
        are ephemeral fixtures — validated on demand via ``check()`` but
        never emitted during class construction.
        """
        if c.ENFORCEMENT_MODE is c.EnforcementMode.OFF:
            return
        if ub.is_function_local(model_type):
            return
        report = FlextUtilitiesEnforcement.check(model_type)
        FlextUtilitiesEnforcement.emit(report)

    @staticmethod
    def run_layer(target: type, layer: str) -> None:
        """Namespace ``__init_subclass__`` hook — run layer + namespace rules.

        Function-local classes skip emission for the same reason as ``run``.
        """
        if c.ENFORCEMENT_NAMESPACE_MODE is c.EnforcementMode.OFF:
            return
        if ub.is_function_local(target):
            return
        report = FlextUtilitiesEnforcement.check(target, layer=layer)
        FlextUtilitiesEnforcement.emit(
            report,
            mode=c.ENFORCEMENT_NAMESPACE_MODE,
        )


__all__: list[str] = ["FlextUtilitiesEnforcement"]
