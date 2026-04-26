"""Annotation inspection + rule-predicate surface for enforcement.

Every row in ``c.ENFORCEMENT_RULES`` is paired by naming convention with a
``check_<tag>`` staticmethod on :class:`ube`.

Every ``check_<tag>`` returns ``t.StrMapping | None``:

* ``None`` → no violation.
* Empty mapping → violation with the default template strings.
* Non-empty mapping → violation; keys are substituted into the rule's
  ``problem`` and ``fix`` templates via ``.format(**mapping)``.

This is the SSOT for per-violation context. Engine renders each violation
with the mapping it received — no string concatenation, no loose types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import functools
import inspect
import re
from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
)
from enum import EnumType
from pathlib import Path
from types import UnionType
from typing import (
    Annotated,
    Any,
    ForwardRef,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    no_type_check,
)

from pydantic.fields import FieldInfo

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cp
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}
type _DefaultFactory = type | Callable[..., tp.JsonValue] | None


@no_type_check
class FlextUtilitiesBeartypeEngine:
    """Annotation inspection + per-tag rule predicates (static-only)."""

    # ------------------------------------------------------------------
    # Low-level annotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def unwrap_type_alias(
        hint: t.TypeHintSpecifier | None,
    ) -> t.TypeHintSpecifier | None:
        current = hint
        seen: set[int] = set()
        while isinstance(current, TypeAliasType):
            current_id = id(current)
            if current_id in seen:
                return current
            seen.add(current_id)
            current = current.__value__
        return current

    @staticmethod
    def contains_any(hint: t.TypeHintSpecifier | None) -> bool:
        return FlextUtilitiesBeartypeEngine._contains_any(hint, seen=set())

    @staticmethod
    def _contains_any(
        hint: t.TypeHintSpecifier | None,
        *,
        seen: set[int],
    ) -> bool:
        hint = ube.unwrap_type_alias(hint)
        if hint is None:
            return False
        hint_id = id(hint)
        if hint_id in seen:
            return False
        seen.add(hint_id)
        if hint is Any or hint is object:
            return True
        return any(
            FlextUtilitiesBeartypeEngine._contains_any(arg, seen=seen)
            for arg in get_args(hint)
        )

    @staticmethod
    def has_forbidden_collection_origin(
        hint: t.TypeHintSpecifier | None,
        forbidden: frozenset[str],
    ) -> tuple[bool, str]:
        hint = ube.unwrap_type_alias(hint)
        if hint is None:
            return False, ""
        origin = get_origin(hint)
        if origin is not None and hasattr(origin, "__name__"):
            name: str = origin.__name__
            if name in forbidden:
                return True, name
        return False, ""

    @staticmethod
    def count_union_members(hint: t.TypeHintSpecifier | None) -> int:
        hint = ube.unwrap_type_alias(hint)
        if hint is None:
            return 0
        origin = get_origin(hint)
        if origin is not UnionType and origin is not Union:
            return 0
        return sum(1 for a in get_args(hint) if a is not type(None))

    @staticmethod
    def matches_str_none_union(hint: t.TypeHintSpecifier | None) -> bool:
        hint = ube.unwrap_type_alias(hint)
        if hint is None:
            return False
        origin = get_origin(hint)
        if origin is not UnionType and origin is not Union:
            return False
        args = get_args(hint)
        return str in args and type(None) in args

    @staticmethod
    def alias_contains_any(
        alias_value: t.TypeHintSpecifier | None,
    ) -> bool:
        try:
            return ube.contains_any(alias_value)
        except (TypeError, AttributeError, RuntimeError, RecursionError):
            return "Any" in str(alias_value)

    @staticmethod
    def unwrap_annotated(
        hint: t.TypeHintSpecifier | None,
    ) -> t.TypeHintSpecifier | None:
        current = hint
        while current is not None:
            current = ube.unwrap_type_alias(current)
            if isinstance(current, ForwardRef):
                current = current.__forward_arg__
                continue
            if isinstance(current, str):
                stripped = current.strip()
                annotated_prefix = next(
                    (
                        prefix
                        for prefix in (
                            "Annotated[",
                            "typing.Annotated[",
                            "typing_extensions.Annotated[",
                        )
                        if stripped.startswith(prefix) and stripped.endswith("]")
                    ),
                    None,
                )
                if annotated_prefix is None:
                    return stripped
                current = FlextUtilitiesBeartypeEngine._first_top_level_arg(
                    stripped.removeprefix(annotated_prefix)[:-1],
                )
                continue
            if get_origin(current) is not Annotated:
                return current
            args = get_args(current)
            if not args:
                return current
            current = args[0]
        return current

    @staticmethod
    def _first_top_level_arg(annotation_text: str) -> str:
        depth = 0
        for index, char in enumerate(annotation_text):
            if char == "[":
                depth += 1
                continue
            if char == "]":
                depth -= 1
                continue
            if char == "," and depth == 0:
                return annotation_text[:index].strip()
        return annotation_text.strip()

    # ------------------------------------------------------------------
    # Rule-local helpers (shared)
    # ------------------------------------------------------------------

    @staticmethod
    def mutable_kind(value: tp.JsonValue) -> str | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if isinstance(value, kind):
                return kind.__name__
        return None

    @staticmethod
    def mutable_default_factory_kind(factory: _DefaultFactory) -> type | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if factory is kind or get_origin(factory) is kind:
                return kind
        return None

    @staticmethod
    def allows_mutable_default_factory(
        hint: t.TypeHintSpecifier | None,
        factory: _DefaultFactory,
    ) -> bool:
        expected_by_factory: Mapping[type, type] = {
            list: MutableSequence,
            dict: MutableMapping,
            set: MutableSet,
        }
        mutable_kind = ube.mutable_default_factory_kind(
            factory,
        )
        if mutable_kind is None:
            return False
        expected = expected_by_factory.get(mutable_kind)
        normalized = ube.unwrap_annotated(hint)
        if normalized is None:
            return False
        if isinstance(normalized, str):
            expected_name = expected.__name__ if expected is not None else ""
            return bool(expected_name) and (
                normalized == expected_name
                or normalized.startswith((
                    f"{expected_name}[",
                    f"typing.{expected_name}[",
                    f"collections.abc.{expected_name}[",
                ))
            )
        origin = get_origin(normalized)
        target = origin or normalized
        return expected is not None and target is expected

    @staticmethod
    def has_relaxed_extra_base(target: type) -> bool:
        return any(
            b.__name__ in c.ENFORCEMENT_RELAXED_EXTRA_BASES for b in target.__mro__
        )

    @staticmethod
    def has_runtime_protocol_marker(value: type) -> bool:
        return bool(getattr(value, "_is_protocol", False))

    @staticmethod
    def has_abstract_contract(value: type) -> bool:
        if getattr(value, "__abstractmethods__", None):
            return True
        return any(getattr(b, "__name__", "") == "ABC" for b in value.__mro__)

    @staticmethod
    def has_nested_namespace(value: type) -> bool:
        """True when ``value`` directly *or via MRO* hosts public inner classes.

        Namespace holders in the protocols layer commonly build their
        inner-class tree through MRO composition (``class Cli(Pipeline, Domain)``
        where parents carry the actual ``Protocol`` subclasses). Walking only
        ``vars(value)`` would miss them and flag a legitimate container.
        """
        for base in value.__mro__:
            if base is object:
                continue
            if any(
                isinstance(v, type) and not n.startswith("_")
                for n, v in vars(base).items()
            ):
                return True
        return False

    @staticmethod
    def defined_inside(inner_cls: type, outer_qualname: str) -> bool:
        """True when inner_cls was actually defined inside outer (not aliased)."""
        qn = getattr(inner_cls, "__qualname__", "")
        return qn.startswith(f"{outer_qualname}.")

    @staticmethod
    def defined_in_function_scope(target: type) -> bool:
        """True when target is defined inside a function body.

        Python marks function-local classes with ``<locals>`` in their
        qualname — a purely runtime signal, no hardcoded name match.
        Ephemeral classes (test fixtures, inner factories) should never
        be held to namespace-facade rules.
        """
        return "<locals>" in getattr(target, "__qualname__", "")

    # ------------------------------------------------------------------
    # FIELD predicates
    # ------------------------------------------------------------------

    @staticmethod
    def check_no_any(info: FieldInfo) -> t.StrMapping | None:
        return (
            _BARE_VIOLATION
            if ube.contains_any(
                info.annotation,
            )
            else _NO_VIOLATION
        )

    @staticmethod
    def check_no_bare_collection(
        info: FieldInfo,
    ) -> t.StrMapping | None:
        bad, origin = ube.has_forbidden_collection_origin(
            info.annotation,
            c.ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS,
        )
        if not bad:
            return _NO_VIOLATION
        replacement = next(
            (
                repl
                for kind, repl in c.ENFORCEMENT_FORBIDDEN_COLLECTIONS.items()
                if kind.__name__ == origin
            ),
            origin,
        )
        return {"kind": origin, "replacement": replacement}

    @staticmethod
    def check_no_mutable_default(
        info: FieldInfo,
    ) -> t.StrMapping | None:
        kind = ube.mutable_kind(info.default)
        if kind is None or not info.default:
            return _NO_VIOLATION
        return {"kind": kind}

    @staticmethod
    def check_no_raw_collections_field_default(
        info: FieldInfo,
    ) -> t.StrMapping | None:
        """Reject Field(default_factory=list/dict/set).

        When a field uses ``default_factory`` pointing to a mutable
        collection constructor, it should use the immutable equivalent
        instead (tuple, MappingProxyType, frozenset).
        """
        factory = info.default_factory
        if factory is None:
            return _NO_VIOLATION
        if ube.allows_mutable_default_factory(
            info.annotation,
            factory,
        ):
            return _NO_VIOLATION
        mutable_kind = ube.mutable_default_factory_kind(
            factory,
        )
        if mutable_kind is not None:
            return {"kind": mutable_kind.__name__}
        return _NO_VIOLATION

    @staticmethod
    def check_no_str_none_empty(
        info: FieldInfo,
    ) -> t.StrMapping | None:
        if not ube.matches_str_none_union(info.annotation):
            return _NO_VIOLATION
        if isinstance(info.default, str) and not info.default:
            return _BARE_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def check_no_inline_union(
        info: FieldInfo,
    ) -> t.StrMapping | None:
        arms = ube.count_union_members(info.annotation)
        if arms <= c.ENFORCEMENT_INLINE_UNION_MAX:
            return _NO_VIOLATION
        return {"arms": str(arms)}

    @staticmethod
    def check_missing_description(
        model_type: type[mp.BaseModel],
        name: str,
        info: FieldInfo,
    ) -> t.StrMapping | None:
        if name.startswith("_") or info.description:
            return _NO_VIOLATION
        raw_ann = vars(model_type).get("__annotations__", {})
        raw = raw_ann.get(name)
        if isinstance(raw, str) and "description=" in raw:
            return _NO_VIOLATION
        resolved = inspect.get_annotations(model_type, eval_str=False)
        ann = resolved.get(name)
        if get_origin(ann) is Annotated:
            for meta in get_args(ann)[1:]:
                if isinstance(meta, FieldInfo) and meta.description:
                    return _NO_VIOLATION
        return _BARE_VIOLATION

    # ------------------------------------------------------------------
    # MODEL_CLASS predicates
    # ------------------------------------------------------------------

    @staticmethod
    def check_no_v1_config(target: type) -> t.StrMapping | None:
        return (
            _BARE_VIOLATION
            if isinstance(
                target.__dict__.get("Config"),
                type,
            )
            else _NO_VIOLATION
        )

    @staticmethod
    def check_extra_missing(target: type) -> t.StrMapping | None:
        if ube.has_relaxed_extra_base(target):
            return _NO_VIOLATION
        if not issubclass(target, mp.BaseModel):
            return _NO_VIOLATION
        return (
            _BARE_VIOLATION
            if target.model_config.get(
                "extra",
            )
            is None
            else _NO_VIOLATION
        )

    @staticmethod
    def check_extra_wrong(target: type) -> t.StrMapping | None:
        if ube.has_relaxed_extra_base(target):
            return _NO_VIOLATION
        if not issubclass(target, mp.BaseModel):
            return _NO_VIOLATION
        extra = target.model_config.get("extra")
        local = target.__dict__.get("model_config", {})
        if extra in {None, "forbid"} or "extra" not in local:
            return _NO_VIOLATION
        return {"extra": str(extra)}

    @staticmethod
    def check_value_not_frozen(target: type) -> t.StrMapping | None:
        if not any(
            b.__name__ in c.ENFORCEMENT_VALUE_OBJECT_BASES for b in target.__mro__
        ):
            return _NO_VIOLATION
        if not issubclass(target, mp.BaseModel):
            return _NO_VIOLATION
        return (
            _BARE_VIOLATION
            if not target.model_config.get(
                "frozen",
                False,
            )
            else _NO_VIOLATION
        )

    # ------------------------------------------------------------------
    # ATTR filters (not predicates — used by engine to pre-filter names)
    # ------------------------------------------------------------------

    @staticmethod
    def attr_accept_constants(
        name: str,
        value: (
            tp.JsonValue
            | type
            | classmethod[type, ..., tp.JsonValue]
            | staticmethod[..., tp.JsonValue]
            | property
        ),
    ) -> bool:
        if name.startswith("_") or name in c.ENFORCEMENT_CONSTANTS_SKIP_ATTRS:
            return False
        if isinstance(value, (type, classmethod, staticmethod, property)):
            return False
        return not callable(value)

    @staticmethod
    def attr_accept_public(name: str) -> bool:
        return not name.startswith("_")

    @staticmethod
    def attr_accept_utility(name: str) -> bool:
        return (
            name not in c.ENFORCEMENT_UTILITIES_EXEMPT_METHODS
            and not name.startswith("_")
        )

    # ------------------------------------------------------------------
    # ATTR predicates
    # ------------------------------------------------------------------

    @staticmethod
    def check_const_mutable(
        name: str,
        value: tp.JsonValue,
    ) -> t.StrMapping | None:
        del name
        kind = ube.mutable_kind(value)
        if kind is None:
            return _NO_VIOLATION
        return {"kind": kind}

    @staticmethod
    def check_const_lowercase(
        name: str,
        value: tp.JsonValue,
    ) -> t.StrMapping | None:
        del value
        return _BARE_VIOLATION if name != name.upper() else _NO_VIOLATION

    @staticmethod
    def check_alias_any(
        name: str,
        value: tp.JsonValue,
    ) -> t.StrMapping | None:
        del name
        if ube.alias_contains_any(
            getattr(value, "__value__", None),
        ):
            return _BARE_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def check_typeadapter_name(
        name: str,
        value: tp.JsonValue,
    ) -> t.StrMapping | None:
        if type(value).__name__ != "TypeAdapter":
            return _NO_VIOLATION
        if name.startswith("ADAPTER_") or name.upper() == name:
            return _NO_VIOLATION
        return {"name": name, "upper_name": name.upper()}

    @staticmethod
    def check_utility_not_static(
        name: str,
        value: (
            tp.JsonValue
            | staticmethod[..., tp.JsonValue]
            | classmethod[type, ..., tp.JsonValue]
            | Callable[..., tp.JsonValue]
        ),
    ) -> t.StrMapping | None:
        del name
        if isinstance(value, (staticmethod, classmethod)):
            return _NO_VIOLATION
        if inspect.isfunction(value):
            return _BARE_VIOLATION
        return _NO_VIOLATION

    # ------------------------------------------------------------------
    # NAMESPACE predicates
    # ------------------------------------------------------------------

    @staticmethod
    def check_class_prefix(
        target: type,
        expected: str,
    ) -> t.StrMapping | None:
        if target.__name__.startswith(expected):
            return _NO_VIOLATION
        return {"expected": expected, "actual": target.__name__}

    @staticmethod
    def check_cross_strenum(
        inner_cls: type,
        layer: str,
    ) -> t.StrMapping | None:
        allowed = c.ENFORCEMENT_LAYER_ALLOWS.get(layer, frozenset())
        if isinstance(inner_cls, EnumType) and "StrEnum" not in allowed:
            return _BARE_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def check_cross_protocol(
        inner_cls: type,
        layer: str,
    ) -> t.StrMapping | None:
        allowed = c.ENFORCEMENT_LAYER_ALLOWS.get(layer, frozenset())
        is_proto = ube.has_runtime_protocol_marker(
            inner_cls,
        )
        if is_proto and "Protocol" not in allowed:
            return _BARE_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def check_nested_mro(
        target: type,
        expected: str,
    ) -> t.StrMapping | None:
        parts = target.__qualname__.split(".")
        # Top-level classes ARE containers (the roots other classes nest
        # into) — they do not themselves need to be nested. Only nested
        # classes must live under an expected-prefixed outer container.
        if len(parts) < c.ENFORCEMENT_NESTED_MRO_MIN_DEPTH:
            return _NO_VIOLATION
        if not parts[0].startswith(expected):
            return {"expected": expected}
        return _NO_VIOLATION

    # ------------------------------------------------------------------
    # PROTOCOL_TREE predicates
    # ------------------------------------------------------------------

    @staticmethod
    def check_proto_inner_kind(value: type) -> t.StrMapping | None:
        if value.__dict__.get("_flext_enforcement_exempt", False):
            return _NO_VIOLATION
        if ube.has_runtime_protocol_marker(value):
            return _NO_VIOLATION
        if ube.has_nested_namespace(value):
            return _NO_VIOLATION
        if ube.has_abstract_contract(value):
            return _NO_VIOLATION
        return _BARE_VIOLATION

    @staticmethod
    def check_proto_not_runtime(value: type) -> t.StrMapping | None:
        if not ube.has_runtime_protocol_marker(value):
            return _NO_VIOLATION
        if getattr(value, "_is_runtime_protocol", False):
            return _NO_VIOLATION
        return _BARE_VIOLATION

    # ------------------------------------------------------------------
    # Method-name / class-inheritance rules (§3.1, §2.6)
    # ------------------------------------------------------------------

    @staticmethod
    def check_no_accessor_methods(
        _target: type,
        name: str,
    ) -> t.StrMapping | None:
        """``get_*`` / ``set_*`` / ``is_*`` public methods are forbidden.

        AGENTS.md §3.1 Accessor Naming Law: expose state as fields or
        ``@u.computed_field``; use domain verbs (``fetch_*``, ``resolve_*``,
        ``compute_*``) for external boundary reads; booleans use noun/adjective
        names (``success``, ``failure``, ``expired``, ...).
        """
        prefixes = (
            ("get_", "fetch_/resolve_/compute_"),
            ("set_", "configure/apply/update or model_copy(update=...)"),
            ("is_", "a noun/adjective (success, expired, connected, ...)"),
        )
        for prefix, suggestion in prefixes:
            if name.startswith(prefix) and not name.startswith("_"):
                return {"name": name, "suggestion": suggestion}
        return _NO_VIOLATION

    @staticmethod
    def check_settings_inheritance(
        target: type,
    ) -> t.StrMapping | None:
        """Top-level ``*Settings`` classes must inherit from ``FlextSettings``.

        AGENTS.md §2.6 Settings Law. Only applies to classes defined at
        module top-level (qualname without ``.``) — classes nested inside
        a ``Flext*Settings`` or ``Flext*Models`` namespace are configuration
        metadata, not independent settings facades.
        """
        if not target.__name__.endswith("Settings"):
            return _NO_VIOLATION
        if "." in target.__qualname__:
            return _NO_VIOLATION
        if target.__name__ == "FlextSettings":
            return _NO_VIOLATION
        for base in target.__mro__[1:]:
            if base.__name__ == "FlextSettings":
                return _NO_VIOLATION
        return {"name": target.__name__}

    # -----------------------------------------------------------------------
    # ENFORCE-039 / ENFORCE-041 / ENFORCE-043 / ENFORCE-044 detection hooks
    # -----------------------------------------------------------------------

    @staticmethod
    @functools.cache
    def _apt_load_ast(target: type) -> tuple[str, ast.Module] | None:
        """Source-skip helper for every A-PT hook.

        Returns (src_file, ast_tree) or ``None`` to signal "skip this target."
        Skip conditions: nested qualname, unavailable source, third-party
        path (``site-packages`` / ``dist-packages`` / system lib), unparseable
        source.

        ``functools.cache`` keys on ``target`` identity so the four hooks
        sharing this helper read+parse a given module once per import session
        instead of four times.
        """
        if "." in target.__qualname__:
            return None
        try:
            src_file = inspect.getfile(target)
        except (OSError, TypeError):
            return None
        if any(marker in src_file for marker in c.ENFORCE_NON_WORKSPACE_PATH_MARKERS):
            return None
        try:
            source = Path(src_file).read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        return src_file, tree

    @staticmethod
    def _apt_load_wrapper_surface(
        target: type,
    ) -> tuple[str, str, ast.Module] | None:
        """Return source for tests/examples/scripts modules that are real scan targets."""
        loaded = FlextUtilitiesBeartypeEngine._apt_load_ast(target)
        if loaded is None:
            return None

        src_file, tree = loaded
        normalized = src_file.replace("\\", "/")
        if not any(
            segment in normalized for segment in ("/tests/", "/examples/", "/scripts/")
        ):
            return None
        if normalized.endswith("/__init__.py"):
            return None
        try:
            source = Path(src_file).read_text(encoding="utf-8")
        except OSError:
            return None
        return src_file, source, tree

    # Pure AST finders — single SSOT for ENFORCE-039/041/043/044 detection.
    # The runtime ``check_<tag>`` wrappers below consume these. Adding or
    # changing a detection rule means editing ONE finder.

    @staticmethod
    def find_cast_calls(tree: ast.Module) -> Iterator[ast.Call]:
        """Yield every ``cast(...)`` call (ENFORCE-039 detection primitive)."""
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == c.EnforceAstHookSymbol.CAST_CALL
            ):
                yield node

    @staticmethod
    def find_model_rebuild_calls(tree: ast.Module) -> Iterator[ast.Call]:
        """Yield every ``X.model_rebuild(...)`` call (ENFORCE-041 detection primitive)."""
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == c.EnforceAstHookSymbol.MODEL_REBUILD_ATTR
            ):
                yield node

    @staticmethod
    def find_pass_through_wrappers(tree: ast.Module) -> Iterator[ast.FunctionDef]:
        """Yield every pass-through wrapper FunctionDef (ENFORCE-043).

        A wrapper is a function whose only body (after stripping a leading
        docstring) is ``return inner(arg1, arg2, ...)`` where ``inner`` is
        called with the wrapper's positional parameters in order, no keyword
        arguments — a pure delegation that should be inlined.
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            body = [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                )
            ]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            ret_value = body[0].value
            if not isinstance(ret_value, ast.Call):
                continue
            param_names = [a.arg for a in node.args.args]
            call_args = [
                a.id if isinstance(a, ast.Name) else None for a in ret_value.args
            ]
            if param_names and param_names == call_args and not ret_value.keywords:
                yield node

    @staticmethod
    def find_private_attr_probes(
        tree: ast.Module,
    ) -> Iterator[tuple[ast.Call, str, str]]:
        """Yield ``(node, builtin_name, attr_name)`` for every private-attr probe (ENFORCE-044).

        Matches ``hasattr/getattr/setattr(obj, "_private")`` where the second
        argument is a string literal beginning with a single underscore (not
        a dunder). Dunder names are legitimate runtime introspection. Yielding
        the already-narrowed ``builtin_name`` and ``attr_name`` strings lets
        every consumer skip re-narrowing the AST nodes.
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            builtin_name = node.func.id
            if builtin_name not in c.ENFORCE_PRIVATE_PROBE_BUILTINS:
                continue
            if len(node.args) < c.ENFORCE_PRIVATE_PROBE_MIN_ARGS or not isinstance(
                node.args[1], ast.Constant
            ):
                continue
            attr_name = node.args[1].value
            if not isinstance(attr_name, str):
                continue
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                yield node, builtin_name, attr_name

    # ``check_<tag>`` wrappers — runtime entrypoint for the ENFORCE-039/041/043/044
    # rules dispatched via ``c.ENFORCEMENT_RULES``. Each wrapper loads the AST,
    # delegates to the matching pure finder, and adapts the first match to the
    # canonical ``t.StrMapping | None`` violation contract.

    @staticmethod
    def check_cast_outside_core(target: type) -> t.StrMapping | None:
        """``cast()`` call outside flext-core result internals (ENFORCE-039)."""
        loaded = FlextUtilitiesBeartypeEngine._apt_load_ast(target)
        if loaded is None:
            return _NO_VIOLATION
        src_file, tree = loaded
        if any(marker in src_file for marker in c.ENFORCE_FLEXT_CORE_PATH_MARKERS):
            return _NO_VIOLATION
        for node in FlextUtilitiesBeartypeEngine.find_cast_calls(tree):
            return {"file": Path(src_file).name, "line": str(node.lineno)}
        return _NO_VIOLATION

    @staticmethod
    def check_model_rebuild_call(target: type) -> t.StrMapping | None:
        """``model_rebuild()`` invocation indicates unresolved forward refs (ENFORCE-041)."""
        loaded = FlextUtilitiesBeartypeEngine._apt_load_ast(target)
        if loaded is None:
            return _NO_VIOLATION
        src_file, tree = loaded
        for node in FlextUtilitiesBeartypeEngine.find_model_rebuild_calls(tree):
            return {"file": Path(src_file).name, "line": str(node.lineno)}
        return _NO_VIOLATION

    @staticmethod
    def check_pass_through_wrapper(target: type) -> t.StrMapping | None:
        """Pass-through wrapper detection (ENFORCE-043)."""
        loaded = FlextUtilitiesBeartypeEngine._apt_load_ast(target)
        if loaded is None:
            return _NO_VIOLATION
        src_file, tree = loaded
        for node in FlextUtilitiesBeartypeEngine.find_pass_through_wrappers(tree):
            return {"name": node.name, "file": Path(src_file).name}
        return _NO_VIOLATION

    @staticmethod
    def check_private_attr_probe(target: type) -> t.StrMapping | None:
        """``hasattr/getattr/setattr`` probing of private attributes (ENFORCE-044)."""
        loaded = FlextUtilitiesBeartypeEngine._apt_load_ast(target)
        if loaded is None:
            return _NO_VIOLATION
        src_file, tree = loaded
        for (
            _node,
            builtin,
            attr,
        ) in FlextUtilitiesBeartypeEngine.find_private_attr_probes(
            tree,
        ):
            return {
                "probe": builtin,
                "name": attr,
                "file": Path(src_file).name,
            }
        return _NO_VIOLATION

    @staticmethod
    def check_no_core_tests_namespace(target: type) -> t.StrMapping | None:
        """Deprecated ``.Core.Tests`` namespace usage in tests/examples/scripts (ENFORCE-054)."""
        loaded = FlextUtilitiesBeartypeEngine._apt_load_wrapper_surface(target)
        if loaded is None:
            return _NO_VIOLATION

        src_file, _source, tree = loaded
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "Tests":
                continue
            parent_attr = node.value if isinstance(node.value, ast.Attribute) else None
            if parent_attr is None or parent_attr.attr != "Core":
                continue
            base_name = (
                parent_attr.value if isinstance(parent_attr.value, ast.Name) else None
            )
            if base_name is None or base_name.id not in cp.RUNTIME_ALIAS_NAMES:
                continue
            return {
                "symbol": f"{base_name.id}.Core.Tests",
                "file": Path(src_file).name,
                "line": str(node.lineno),
            }
        return _NO_VIOLATION

    @staticmethod
    def check_no_wrapper_root_alias_import(target: type) -> t.StrMapping | None:
        """Wrapper aliases must be imported from root package in tests/examples/scripts (ENFORCE-055)."""
        loaded = FlextUtilitiesBeartypeEngine._apt_load_wrapper_surface(target)
        if loaded is None:
            return _NO_VIOLATION

        src_file, source, tree = loaded

        wrapper_submodules = cp.FACADE_MODULE_NAMES
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module_name = node.module or ""
            parent, dot, child = module_name.partition(".")
            if (
                not dot
                or parent not in {"tests", "examples", "scripts"}
                or child not in wrapper_submodules
            ):
                continue
            if not any(
                (alias.asname or alias.name) in cp.RUNTIME_ALIAS_NAMES
                for alias in node.names
            ):
                continue
            statement = ast.get_source_segment(source, node) or (
                f"from {module_name} import "
                + ", ".join(
                    alias.name
                    if alias.asname is None
                    else f"{alias.name} as {alias.asname}"
                    for alias in node.names
                )
            )
            return {
                "file": Path(src_file).name,
                "line": str(node.lineno),
                "statement": statement.strip(),
            }
        return _NO_VIOLATION

    # -----------------------------------------------------------------------
    # R1–R10 MRO compliance checks
    # -----------------------------------------------------------------------

    @staticmethod
    def check_no_concrete_namespace_import(
        target: type,
    ) -> t.StrMapping | None:
        """R1, R3: Reject bare Flext* class imports in canonical files.

        Canonical files (typings, models, protocols, utilities, constants)
        must import ONLY aliases (c, m, t, u, p) from parent, never concrete
        ``FlextXxxYyy`` classes. Only permitted exception: Pattern-B utilities
        must import FlextPeerXxx concrete class for second parent.
        """
        try:
            src_file = inspect.getfile(target)
            filename = Path(src_file).name
            if filename not in c.ENFORCEMENT_CANONICAL_FILES:
                return _NO_VIOLATION

            source = Path(src_file).read_text(encoding="utf-8")

            flext_imports = re.findall(
                r"from\s+(flext_\w+)\s+import\s+([^#\n]+)", source
            )
            for _module, imports_str in flext_imports:
                imports = [i.strip() for i in imports_str.split(",")]
                for imp in imports:
                    if imp.startswith("Flext") and " as " not in imp:
                        return {"file": filename, "import": imp}

            return _NO_VIOLATION
        except (TypeError, OSError, AttributeError):
            return _NO_VIOLATION

    @staticmethod
    def check_no_pydantic_consumer_import(
        target: type,
    ) -> t.StrMapping | None:
        """R2: Reject bare pydantic imports in consumers.

        Only flext-core._* modules and flext-*._* base pyramids may import
        from pydantic. All consumer code must use u.*, m.* facades from parent.
        """
        try:
            module_name = getattr(target, "__module__", "") or ""
            if module_name.startswith("flext_core._"):
                return _NO_VIOLATION
            if any(
                module_name.startswith(f"{pkg}._")
                for pkg in [
                    "flext_cli",
                    "flext_web",
                    "flext_meltano",
                    "flext_ldap",
                    "flext_api",
                    "flext_auth",
                    "flext_infra",
                    "flext_tests",
                    "flext_observability",
                ]
            ):
                return _NO_VIOLATION

            src_file = inspect.getfile(target)
            source = Path(src_file).read_text(encoding="utf-8")

            pydantic_imports = re.findall(
                r"from\s+pydantic\s+import\s+([^#\n]+)",
                source,
            )
            forbidden = {
                "BaseModel",
                "Field",
                "ConfigDict",
                "TypeAdapter",
                "field_validator",
                "model_validator",
                "computed_field",
                "PrivateAttr",
                "AfterValidator",
                "BeforeValidator",
            }
            for import_str in pydantic_imports:
                imports = [i.strip() for i in import_str.split(",")]
                for imp in imports:
                    name = imp.split(" as ")[0].strip()
                    if name in forbidden:
                        return {"import": name, "package": module_name.split(".")[0]}

            return _NO_VIOLATION
        except (TypeError, OSError, AttributeError):
            return _NO_VIOLATION

    @staticmethod
    def check_facade_base_is_alias_or_peer(
        target: type,
    ) -> t.StrMapping | None:
        """R4, R5: Facade class bases must be alias or peer concrete class.

        Pattern A: class FlextXxxTypes(t):
        Pattern B: class FlextQualityTypes(t, FlextWebTypes):
        """
        if not target.__bases__:
            return _NO_VIOLATION

        first_base = target.__bases__[0]
        first_name = getattr(first_base, "__name__", "")

        if first_name in {"t", "m", "p", "c", "u", "r", "s", "x", "d", "e", "h"}:
            return _NO_VIOLATION

        if first_name.startswith("Flext"):
            return {"base": first_name, "expected": "alias or FlextPeerXxx"}

        return _NO_VIOLATION

    @staticmethod
    def check_alias_first_multi_parent(
        target: type,
    ) -> t.StrMapping | None:
        """R5: Multi-parent facades must have alias as first base.

        MRO C3 linearization requires alias (FlextCliTypes) before peer
        (FlextWebTypes) to avoid ambiguity resolution failures.
        """
        min_multi_parent = 2
        if len(target.__bases__) < min_multi_parent:
            return _NO_VIOLATION

        first_base = target.__bases__[0]
        first_name = getattr(first_base, "__name__", "")

        aliases = {"t", "m", "p", "c", "u", "r", "s", "x", "d", "e", "h"}
        if first_name not in aliases:
            return {"bases": str(len(target.__bases__)), "first": first_name}

        return _NO_VIOLATION

    @staticmethod
    def check_alias_rebound_at_module_end(
        target: type,
    ) -> t.StrMapping | None:
        """R6: Module must rebind the canonical alias at end-of-file.

        After class definition, module must end with: t = FlextXxxTypes
        to establish the public contract.
        """
        try:
            src_file = inspect.getfile(target)
            filename = Path(src_file).name
            if filename not in c.ENFORCEMENT_CANONICAL_FILES:
                return _NO_VIOLATION

            source = Path(src_file).read_text(encoding="utf-8")
            target_name = target.__name__
            alias_char: str | None = None
            match target_name:
                case value if "Types" in value:
                    alias_char = "t"
                case value if "Models" in value:
                    alias_char = "m"
                case value if "Protocols" in value:
                    alias_char = "p"
                case value if "Constants" in value:
                    alias_char = "c"
                case value if "Utilities" in value:
                    alias_char = "u"
                case _:
                    alias_char = None

            violation: t.StrMapping | None = _NO_VIOLATION
            if alias_char:
                last_assign = next(
                    (
                        line.strip()
                        for line in reversed(source.strip().split("\n"))
                        if "=" in line
                    ),
                    "",
                )
                expected = f"{alias_char} = {target_name}"
                if expected not in last_assign:
                    violation = {"alias": alias_char, "class": target_name}
            return violation
        except (TypeError, OSError, AttributeError):
            return _NO_VIOLATION

    @staticmethod
    def check_no_redundant_inner_namespace(
        target: type,
    ) -> t.StrMapping | None:
        """R8: No redundant inner namespace re-inheritance.

        If parent already exposes a namespace (e.g., t.Cli from FlextCliTypes),
        the child must NOT redefine it locally with empty body.
        """
        if "." not in target.__qualname__:
            return _NO_VIOLATION

        source_code = None
        try:
            src_file = inspect.getfile(target)
            source_code = Path(src_file).read_text(encoding="utf-8")
        except (TypeError, OSError, AttributeError):
            return _NO_VIOLATION

        if not source_code:
            return _NO_VIOLATION

        outer_name = target.__qualname__.split(".")[0]
        inner_name = target.__name__
        pattern = rf"class\s+{inner_name}\s*\([^)]*{outer_name}[^)]*\):\s*pass\s*$"

        if re.search(pattern, source_code, re.MULTILINE):
            return {"class": target.__qualname__}

        return _NO_VIOLATION

    @staticmethod
    def check_no_self_root_import_in_core_files(
        target: type,
    ) -> t.StrMapping | None:
        """R7: Canonical files must not import aliases from own package.

        typings.py, models.py, protocols.py, utilities.py, constants.py must
        import aliases from the PARENT package, never from their own package.
        This prevents circular initialization during lazy loading.
        """
        try:
            src_file = inspect.getfile(target)
            filename = Path(src_file).name
            if filename not in c.ENFORCEMENT_CANONICAL_FILES:
                return _NO_VIOLATION

            module_name = getattr(target, "__module__", "") or ""
            package = module_name.split(".")[0]

            source = Path(src_file).read_text(encoding="utf-8")

            from_imports = re.findall(
                rf"from\s+{re.escape(package)}\s+import\s+([cmptur])\b",
                source,
            )

            if from_imports:
                return {"package": package, "alias": from_imports[0]}

            return _NO_VIOLATION
        except (TypeError, OSError, AttributeError):
            return _NO_VIOLATION

    @staticmethod
    def check_sibling_models_type_checking(
        target: type,
    ) -> t.StrMapping | None:
        """R9: Sibling _models/* imports used only in annotations go under TYPE_CHECKING.

        If a class from _models/sibling.py is referenced only in an Annotated[...],
        that import must be guarded by `if TYPE_CHECKING:`.
        """
        try:
            src_file = inspect.getfile(target)
            if "_models" not in src_file:
                return _NO_VIOLATION

            source = Path(src_file).read_text(encoding="utf-8")

            if "if TYPE_CHECKING:" not in source:
                return _NO_VIOLATION

            non_type_checking_section = source.split("if TYPE_CHECKING:")[0]
            type_checking_section = (
                source.split("if TYPE_CHECKING:")[1]
                if len(source.split("if TYPE_CHECKING:")) > 1
                else ""
            )

            non_tc_imports = re.findall(
                r"from\s+\.(\w+)\s+import\s+([^#\n]+)",
                non_type_checking_section,
            )

            for module, imports_str in non_tc_imports:
                imports = [i.strip() for i in imports_str.split(",")]
                for imp in imports:
                    name = imp.split(" as ")[0].strip()
                    if not name:
                        continue

                    used_in_annotation = re.search(
                        rf"\bAnnotated\s*\[\s*[^]]*\b{re.escape(name)}\b",
                        source,
                    )
                    if not used_in_annotation:
                        continue

                    in_type_checking = re.search(
                        rf"from\s+\.{re.escape(module)}\s+import.*\b{re.escape(name)}\b",
                        type_checking_section,
                    )
                    if not in_type_checking:
                        return {"import": name, "module": module}

            return _NO_VIOLATION
        except (TypeError, OSError, AttributeError, IndexError):
            return _NO_VIOLATION

    @staticmethod
    def check_utilities_explicit_class_when_self_ref(
        target: type,
    ) -> t.StrMapping | None:
        """R10: utilities.py multi-parent must use explicit class base, not alias.

        When a utilities.py method calls u.method() and u is rebound to the
        local class, pyrefly requires the first base to be the explicit parent
        class name, not the alias, to properly resolve the MRO.
        """
        try:
            src_file = inspect.getfile(target)
            if not src_file.endswith("utilities.py"):
                return _NO_VIOLATION

            module_name = getattr(target, "__module__", "") or ""
            package = module_name.split(".")[0]

            if package in c.ENFORCEMENT_PATTERN_B_UTILITIES_WHITELIST:
                return _NO_VIOLATION

            min_multi_parent = 2
            if len(target.__bases__) < min_multi_parent:
                return _NO_VIOLATION

            first_base = target.__bases__[0]
            first_name = getattr(first_base, "__name__", "")

            if first_name == "u":
                source = Path(src_file).read_text(encoding="utf-8")

                if re.search(r"\bu\.(\w+)\s*\(", source):
                    return {
                        "class": target.__name__,
                        "first_base": "u",
                    }

            return _NO_VIOLATION
        except (TypeError, OSError, AttributeError):
            return _NO_VIOLATION


ube = FlextUtilitiesBeartypeEngine
__all__: list[str] = ["FlextUtilitiesBeartypeEngine", "ube"]
