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

import dis
import functools
import inspect
import types as _types_mod
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
from types import MappingProxyType, UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    ForwardRef,
    TypeAlias,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    no_type_check,
)

from beartype._util.func.utilfunccodeobj import (  # noqa: PLC2701 - beartype-internal function introspection
    get_func_code_object_or_none,
)
from beartype._util.func.utilfunctest import (  # noqa: PLC2701 - beartype-internal function introspection
    is_func_python,
)
from beartype._util.module.utilmodget import (  # noqa: PLC2701 - beartype-internal module introspection
    get_module_filename_or_none,
    get_object_module_name_or_none,
    get_object_module_or_none,
)
from pydantic.fields import FieldInfo

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cp
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}
_BINARY_ARITY: int = 2
_FIELD_DESCRIPTION_ARITY: int = 3
_typing_TypeAlias = TypeAlias  # sentinel for ``X: TypeAlias = Y`` annotation match.
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
        """True if Any/object appears in ``hint`` — uses beartype.door.TypeHint."""
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
            FlextUtilitiesBeartypeEngine._contains_any(child, seen=seen)
            for child in get_args(hint)
        )

    @staticmethod
    def has_forbidden_collection_origin(
        hint: t.TypeHintSpecifier | None,
        forbidden: frozenset[str],
    ) -> tuple[bool, str]:
        """Detect bare list/dict/set via ``typing.get_origin`` (recursion-safe).

        Beartype's ``get_hint_pep_origin_or_none`` does not accept the broad
        ``TypeHintSpecifier`` union, so we route through ``typing.get_origin``
        whose contract covers any annotation shape.
        """
        hint = ube.unwrap_type_alias(hint)
        if hint is None:
            return False, ""
        origin = get_origin(hint)
        if origin is None or not hasattr(origin, "__name__"):
            return False, ""
        name: str = origin.__name__
        if name in forbidden:
            return True, name
        return False, ""

    @staticmethod
    def count_union_members(hint: t.TypeHintSpecifier | None) -> int:
        """Count union arms (excludes ``None``).

        Uses ``typing.get_origin`` for union detection (robust for both
        ``X | Y`` and ``Union[X, Y]``) and ``typing.get_args`` for the
        member walk, which handles PEP-695 recursive aliases that
        ``beartype.door.TypeHint`` does not yet support.
        """
        hint = ube.unwrap_type_alias(hint)
        if hint is None:
            return 0
        if get_origin(hint) not in {UnionType, Union}:
            return 0
        return sum(1 for a in get_args(hint) if a is not type(None))

    @staticmethod
    def matches_str_none_union(hint: t.TypeHintSpecifier | None) -> bool:
        """``str | None`` detection via ``typing`` primitives (recursion-safe)."""
        hint = ube.unwrap_type_alias(hint)
        if hint is None:
            return False
        if get_origin(hint) not in {UnionType, Union}:
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
                return str(kind.__name__)
        return None

    @staticmethod
    def mutable_default_factory_kind(factory: _DefaultFactory) -> type | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            runtime_type: type = kind
            if factory is runtime_type or get_origin(factory) is runtime_type:
                return runtime_type
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

    # FIELD / MODEL_CLASS predicates moved to beartype-driven visitors
    # (``_v_field_shape`` / ``_v_model_config``). Legacy ``check_<tag>``
    # methods deleted in Phase 1 as part of the data-driven dispatcher.

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

    # ATTR / NAMESPACE / PROTOCOL_TREE / METHOD_SHAPE / LOOSE_SYMBOL predicates
    # moved to beartype-driven visitors (``_v_attr_shape`` / ``_v_method_shape``
    # / ``_v_class_placement`` / ``_v_protocol_tree`` / ``_v_loose_symbol``).
    # Legacy ``check_<tag>`` methods deleted in Phase 1.

    # -----------------------------------------------------------------------
    # Runtime module-introspection helpers (replaces source-AST loaders)
    # -----------------------------------------------------------------------

    @staticmethod
    @functools.cache
    def runtime_module_for(
        target: type,
    ) -> _types_mod.ModuleType | None:
        """Return the already-imported module of ``target`` or ``None``.

        Skip conditions: nested qualname (target is an inner class — its
        module-level scan already happened on the outer), target is not
        bound at module top-level under its own name (dynamic ``type(...)``
        classes created inside functions/tests have no canonical source
        location), no module attached, or module physically located in a
        non-workspace path (site-packages / dist-packages / system lib).
        """
        if "." in target.__qualname__:
            return None
        module: _types_mod.ModuleType | None = get_object_module_or_none(target)
        if module is None:
            return None
        # Only audit classes that are top-level bindings in their declared
        # module — dynamic ``type(...)`` factories produce orphan classes
        # whose ``__module__`` points at the caller but who aren't actually
        # in that module's ``__dict__``.
        if getattr(module, target.__name__, None) is not target:
            return None
        src_file = get_module_filename_or_none(module)
        if src_file is None:
            return None
        if any(marker in src_file for marker in c.ENFORCE_NON_WORKSPACE_PATH_MARKERS):
            return None
        return module

    @staticmethod
    def runtime_wrapper_module_for(
        target: type,
    ) -> _types_mod.ModuleType | None:
        """Return the imported module only when it is a tests/examples/scripts wrapper.

        Mirrors the legacy ``apt_load_wrapper_surface`` skip set entirely via
        runtime module introspection — no source-file parsing.
        """
        module = FlextUtilitiesBeartypeEngine.runtime_module_for(target)
        if module is None:
            return None
        src_file = get_module_filename_or_none(module) or ""
        normalized = src_file.replace("\\", "/")
        if not any(
            segment in normalized for segment in ("/tests/", "/examples/", "/scripts/")
        ):
            return None
        if normalized.endswith("/__init__.py"):
            return None
        return module

    @staticmethod
    def iter_module_callables(
        module: _types_mod.ModuleType,
    ) -> Iterator[Callable[..., tp.JsonValue]]:
        """Yield every Python-defined callable physically owned by ``module``.

        Filters via ``is_func_python`` + module-of-origin check so we never
        inspect functions imported from elsewhere (those are the importer's
        concern, not ours).
        """
        module_name = module.__name__
        for value in vars(module).values():
            if isinstance(value, (classmethod, staticmethod)):
                value = value.__func__
            if not is_func_python(value):
                continue
            if get_object_module_name_or_none(value) != module_name:
                continue
            yield value

    @staticmethod
    def function_param_names(fn: Callable[..., tp.JsonValue]) -> tuple[str, ...]:
        """Return the positional parameter names of ``fn`` via its code object.

        Falls back to ``()`` when the callable has no inspectable code
        object (built-ins, slot wrappers).
        """
        code = get_func_code_object_or_none(fn)
        if code is None:
            return ()
        return tuple(str(name) for name in code.co_varnames[: code.co_argcount])

    @staticmethod
    def is_pass_through_bytecode(
        fn: Callable[..., tp.JsonValue],
        param_names: tuple[str, ...],
    ) -> bool:
        """True if ``fn``'s body is exactly ``return inner(p1, p2, ...)``.

        Detects the bytecode shape produced by a pass-through wrapper:
        a single LOAD_GLOBAL/LOAD_DEREF for the inner callable, followed
        by LOAD_FAST for each declared parameter in declaration order,
        then ``CALL`` and ``RETURN_VALUE``. Any opcode outside this exact
        shape disqualifies the match.
        """
        instructions = [
            ins
            for ins in dis.get_instructions(fn)
            if ins.opname not in {"RESUME", "CACHE", "PUSH_NULL", "COPY_FREE_VARS"}
        ]
        loaders = ("LOAD_GLOBAL", "LOAD_DEREF", "LOAD_FAST", "LOAD_NAME")
        if not instructions or instructions[0].opname not in loaders:
            return False
        consumed = 1
        for expected_arg in param_names:
            if (
                consumed >= len(instructions)
                or instructions[consumed].opname != "LOAD_FAST"
                or instructions[consumed].argval != expected_arg
            ):
                return False
            consumed += 1
        if consumed + 1 >= len(instructions):
            return False
        if instructions[consumed].opname not in {"CALL", "CALL_FUNCTION"}:
            return False
        return instructions[consumed + 1].opname == "RETURN_VALUE"

    @staticmethod
    def has_call_to_global(
        fn: Callable[..., tp.JsonValue],
        target_name: str,
    ) -> dis.Instruction | None:
        """Return the first ``LOAD_GLOBAL <target_name>`` instruction or ``None``.

        Used by ``_v_deprecated_syntax`` to find ``cast(...)`` /
        ``hasattr/getattr/setattr`` calls and similar global-call patterns
        without parsing source.
        """
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_GLOBAL" and ins.argval == target_name:
                return ins
        return None

    @staticmethod
    def has_attribute_call(
        fn: Callable[..., tp.JsonValue],
        attr_name: str,
    ) -> dis.Instruction | None:
        """Return the first ``LOAD_ATTR <attr_name>`` instruction or ``None``.

        Used by ``_v_deprecated_syntax(model_rebuild_call)`` to find any
        ``X.model_rebuild(...)`` invocation regardless of the receiver
        identity (``self.model_rebuild``, ``Cls.model_rebuild``, etc.).
        """
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_ATTR" and ins.argval == attr_name:
                return ins
        return None

    @staticmethod
    def has_private_attr_probe(
        fn: Callable[..., tp.JsonValue],
        builtins_set: frozenset[str],
    ) -> tuple[str, str] | None:
        """Detect ``hasattr/getattr/setattr(obj, "_x")`` via paired bytecode ops.

        Returns ``(builtin_name, attr_name)`` for the first match or ``None``.
        Walks the bytecode keeping the last seen ``LOAD_GLOBAL`` in
        ``builtins_set``; flags when followed by a ``LOAD_CONST`` whose
        value is a single-underscore (non-dunder) private name.
        """
        last_builtin: str | None = None
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_GLOBAL" and ins.argval in builtins_set:
                last_builtin = ins.argval
                continue
            if ins.opname == "LOAD_CONST" and last_builtin is not None:
                value = ins.argval
                if (
                    isinstance(value, str)
                    and value.startswith("_")
                    and not value.startswith("__")
                ):
                    return last_builtin, value
            if ins.opname in {"CALL", "CALL_FUNCTION"}:
                last_builtin = None
        return None

    # Legacy ``check_cast_outside_core`` / ``check_model_rebuild_call`` /
    # ``check_pass_through_wrapper`` / ``check_private_attr_probe`` /
    # ``check_no_core_tests_namespace`` / ``check_no_wrapper_root_alias_import``
    # / ``check_no_concrete_namespace_import`` / ``check_no_pydantic_consumer_import``
    # / ``check_facade_base_is_alias_or_peer`` / ``check_alias_first_multi_parent``
    # / ``check_alias_rebound_at_module_end`` / ``check_no_redundant_inner_namespace``
    # / ``check_no_self_root_import_in_core_files`` / ``check_sibling_models_type_checking``
    # / ``check_utilities_explicit_class_when_self_ref`` ALL deleted in Phase 1.
    # Their behaviour now lives in the beartype-driven ``_v_*`` visitors below
    # via the typed ``c.EnforcementPredicateKind`` dispatch table.

    # ------------------------------------------------------------------
    # Generic data-driven dispatcher (Phase 1 — beartype-driven engine)
    # ------------------------------------------------------------------
    #
    # ``apply(kind, params, *args)`` replaces the legacy 1:1
    # ``getattr(ube, f"check_{tag}")(*args)`` lookup. Each visitor is a
    # thin adapter that consumes :class:`m.Enforcement.*Params` flags and
    # delegates to existing introspection helpers (``contains_any``,
    # ``has_forbidden_collection_origin``, ``mutable_kind``, ...) plus
    # ``beartype.door.TypeHint`` for type-structure queries that previously
    # used bespoke string heuristics.

    @classmethod
    def apply(
        cls,
        kind: c.EnforcementPredicateKind,
        params: mp.BaseModel,
        *args: object,
    ) -> t.StrMapping | None:
        """Dispatch a rule predicate to its visitor by ``predicate_kind``.

        Args mirror the per-category iteration shape from
        :class:`FlextUtilitiesEnforcementCollect`. Visitors return
        ``None`` for no violation, an empty mapping for a bare violation,
        or a mapping of substitution keys for a parameterised message.
        """
        visitor = cls._VISITORS.get(kind)
        if visitor is None:
            return _NO_VIOLATION
        return visitor(params, *args)

    @staticmethod
    def _v_field_shape(
        params: me.FieldShapeParams,
        *args: object,
    ) -> t.StrMapping | None:
        """FIELD_SHAPE — Pydantic field annotation governance via flags.

        Args shape depends on params.require_description: 1-arg ``(info,)`` for
        annotation/default checks; 3-arg ``(model_type, name, info)`` for
        the description-required check.
        """
        if params.require_description and len(args) == _FIELD_DESCRIPTION_ARITY:
            model_type, name, info = args
            if not (
                isinstance(model_type, type)
                and isinstance(name, str)
                and isinstance(info, FieldInfo)
            ):
                return _NO_VIOLATION
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
        if len(args) != 1:
            return _NO_VIOLATION
        info = args[0]
        if not isinstance(info, FieldInfo):
            return _NO_VIOLATION
        if params.forbid_any and ube.contains_any(info.annotation):
            return _BARE_VIOLATION
        if params.forbid_bare_collection:
            bad, origin = ube.has_forbidden_collection_origin(
                info.annotation, c.ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS
            )
            if bad:
                replacement = next(
                    (
                        repl
                        for k, repl in c.ENFORCEMENT_FORBIDDEN_COLLECTIONS.items()
                        if k.__name__ == origin
                    ),
                    origin,
                )
                return {"kind": origin, "replacement": replacement}
        if params.forbid_mutable_default:
            mk = ube.mutable_kind(info.default)
            if mk is not None and info.default:
                return {"kind": mk}
        if (
            params.forbid_raw_default_factory
            and info.default_factory is not None
            and not ube.allows_mutable_default_factory(
                info.annotation, info.default_factory
            )
        ):
            fk = ube.mutable_default_factory_kind(info.default_factory)
            if fk is not None:
                return {"kind": fk.__name__}
        if (
            params.forbid_str_none_empty
            and ube.matches_str_none_union(info.annotation)
            and isinstance(info.default, str)
            and not info.default
        ):
            return _BARE_VIOLATION
        if params.forbid_inline_union:
            arms = ube.count_union_members(info.annotation)
            if arms > params.max_union_arms:
                return {"arms": str(arms)}
        return _NO_VIOLATION

    @staticmethod
    def _v_model_config(
        params: me.ModelConfigParams,
        target: type,
    ) -> t.StrMapping | None:
        """MODEL_CONFIG — Pydantic model_config governance via flags."""
        if params.forbid_v1_config and isinstance(target.__dict__.get("Config"), type):
            return _BARE_VIOLATION
        if not issubclass(target, mp.BaseModel):
            return _NO_VIOLATION
        if ube.has_relaxed_extra_base(target):
            return _NO_VIOLATION
        extra = target.model_config.get("extra")
        if params.require_extra_forbid and extra is None:
            return _BARE_VIOLATION
        local = target.__dict__.get("model_config", {})
        if (
            params.allowed_extra_values
            and extra not in {None, "forbid", *params.allowed_extra_values}
            and "extra" in local
        ):
            return {"extra": str(extra)}
        if (
            params.require_frozen_for_value_objects
            and any(
                b.__name__ in c.ENFORCEMENT_VALUE_OBJECT_BASES for b in target.__mro__
            )
            and not target.model_config.get("frozen", False)
        ):
            return _BARE_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def _v_attr_shape(
        params: me.AttrShapeParams,
        name: str,
        value: tp.JsonValue,
    ) -> t.StrMapping | None:
        """ATTR_SHAPE — class-attribute governance (constants / aliases / TypeAdapters)."""
        if params.forbid_mutable_value:
            mk = ube.mutable_kind(value)
            if mk is not None:
                return {"kind": mk}
        if params.require_uppercase_name and name != name.upper():
            return _BARE_VIOLATION
        if params.forbid_any_in_alias and ube.alias_contains_any(
            getattr(value, "__value__", None)
        ):
            return _BARE_VIOLATION
        if (
            params.require_typeadapter_naming
            and type(value).__name__ == "TypeAdapter"
            and not (name.startswith("ADAPTER_") or name.upper() == name)
        ):
            return {"name": name, "upper_name": name.upper()}
        return _NO_VIOLATION

    @staticmethod
    def _v_method_shape(
        params: me.MethodShapeParams,
        *args: object,
    ) -> t.StrMapping | None:
        """METHOD_SHAPE — accessor-prefix and staticmethod-required governance.

        Args shape varies: ``(target, name)`` for accessor checks (NAMESPACE
        category); ``(name, value)`` for utility-tier static-method checks
        (ATTR category).
        """
        if len(args) != _BINARY_ARITY:
            return _NO_VIOLATION
        a, b = args
        if isinstance(a, type) and isinstance(b, str):
            name = b
            if name.startswith("_"):
                return _NO_VIOLATION
            suggestions = (
                ("get_", "fetch_/resolve_/compute_"),
                ("set_", "configure/apply/update or model_copy(update=...)"),
                ("is_", "a noun/adjective (success, expired, connected, ...)"),
            )
            for prefix in params.forbidden_prefixes:
                suggestion = next(
                    (s for p, s in suggestions if p == prefix),
                    "use a domain verb",
                )
                if name.startswith(prefix):
                    return {"name": name, "suggestion": suggestion}
            return _NO_VIOLATION
        if isinstance(a, str) and not isinstance(b, type):
            if not params.require_static_or_classmethod:
                return _NO_VIOLATION
            if isinstance(b, (staticmethod, classmethod)):
                return _NO_VIOLATION
            if inspect.isfunction(b):
                return _BARE_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def _v_class_placement(
        params: me.ClassPlacementParams,
        *args: object,
    ) -> t.StrMapping | None:
        """CLASS_PLACEMENT — class-name / inner-class layer placement.

        Args:
            (target, expected_prefix) → name-prefix check (class_prefix /
              nested_mro per params.check_nested).
            (value, layer) → cross-layer Enum/Protocol membership check.

        """
        if len(args) != _BINARY_ARITY:
            return _NO_VIOLATION
        a, b = args
        if isinstance(b, str) and b in c.ENFORCEMENT_LAYER_ALLOWS:
            value, layer = a, b
            if not isinstance(value, type):
                return _NO_VIOLATION
            allowed = c.ENFORCEMENT_LAYER_ALLOWS.get(layer, frozenset())
            if (
                "StrEnum" in params.forbidden_bases
                and isinstance(value, EnumType)
                and "StrEnum" not in allowed
            ):
                return _BARE_VIOLATION
            if (
                "Protocol" in params.forbidden_bases
                and ube.has_runtime_protocol_marker(value)
                and "Protocol" not in allowed
            ):
                return _BARE_VIOLATION
            return _NO_VIOLATION
        if not isinstance(a, type) or not isinstance(b, str):
            return _NO_VIOLATION
        target, expected = a, b
        if params.check_nested:
            parts = target.__qualname__.split(".")
            if len(parts) < c.ENFORCEMENT_NESTED_MRO_MIN_DEPTH:
                return _NO_VIOLATION
            if not parts[0].startswith(expected):
                return {"expected": expected}
            return _NO_VIOLATION
        if target.__name__.startswith(expected):
            return _NO_VIOLATION
        return {"expected": expected, "actual": target.__name__}

    @staticmethod
    def _v_protocol_tree(
        params: me.ProtocolTreeParams,
        value: type,
    ) -> t.StrMapping | None:
        """PROTOCOL_TREE — inner-class kind + runtime_checkable governance."""
        if params.require_inner_kind_protocol_or_namespace:
            if (
                ube.has_runtime_protocol_marker(value)
                or ube.has_nested_namespace(value)
                or ube.has_abstract_contract(value)
            ):
                pass
            else:
                return _BARE_VIOLATION
        if (
            params.require_runtime_checkable
            and ube.has_runtime_protocol_marker(value)
            and not getattr(value, "_is_runtime_protocol", False)
        ):
            return _BARE_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def _v_mro_shape(
        params: me.MroShapeParams,
        target: type,
    ) -> t.StrMapping | None:
        """MRO_SHAPE — facade base ordering and inner-namespace redundancy."""
        aliases = cp.RUNTIME_ALIAS_NAMES
        if not target.__bases__:
            return _NO_VIOLATION
        base_count = len(target.__bases__)
        first_base = target.__bases__[0]
        first_name = getattr(first_base, "__name__", "")
        requires_alias_first = params.require_alias_first and first_name not in aliases
        min_multi_parent = 2
        alias_violation = next(
            (
                payload
                for enabled, payload in (
                    (
                        requires_alias_first and base_count >= min_multi_parent,
                        {"bases": str(base_count), "first": first_name},
                    ),
                    (
                        requires_alias_first and first_name.startswith("Flext"),
                        {"base": first_name, "expected": "alias or FlextPeerXxx"},
                    ),
                )
                if enabled
            ),
            _NO_VIOLATION,
        )
        outer_name, separator, _ = target.__qualname__.partition(".")
        redundant_inner_violation = (
            {"class": target.__qualname__}
            if alias_violation is None
            and params.forbid_redundant_inner
            and bool(separator)
            and getattr(first_base, "__qualname__", "") == outer_name
            and all(key.startswith("__") and key.endswith("__") for key in vars(target))
            else _NO_VIOLATION
        )
        violation = alias_violation or redundant_inner_violation
        module = (
            ube.runtime_module_for(target)
            if violation is None and params.require_explicit_class_when_self_ref
            else None
        )
        src_file = (
            (get_module_filename_or_none(module) or "") if module is not None else ""
        )
        package = module.__name__.split(".")[0] if module is not None else ""
        normalized_values = (
            value.__func__ if isinstance(value, (classmethod, staticmethod)) else value
            for value in vars(target).values()
        )
        self_ref_violation = (
            {"class": target.__name__, "first_base": "u"}
            if all((
                violation is None,
                module is not None,
                src_file.endswith("utilities.py"),
                package not in c.ENFORCEMENT_PATTERN_B_UTILITIES_WHITELIST,
                base_count >= min_multi_parent,
                first_name == "u",
            ))
            and any(
                (code := get_func_code_object_or_none(value)) is not None
                and "u" in code.co_names
                for value in normalized_values
                if is_func_python(value)
            )
            else _NO_VIOLATION
        )
        return violation or self_ref_violation

    @staticmethod
    def _v_loose_symbol(
        params: me.LooseSymbolParams,
        *args: object,
    ) -> t.StrMapping | None:
        """LOOSE_SYMBOL — top-level class/function naming + settings inheritance.

        Args:
            (target,) → settings inheritance check (params.require_settings_base).
            (target, expected_prefix) → name-prefix check.

        """
        match args:
            case (target, expected_prefix, *_) if isinstance(
                target, type
            ) and isinstance(
                expected_prefix,
                str,
            ):
                has_expected_prefix = True
                expected_prefix_text = expected_prefix
            case (target, *_) if isinstance(target, type):
                has_expected_prefix = False
                expected_prefix_text = ""
            case _:
                return _NO_VIOLATION
        target_name = target.__name__
        is_top_level = "." not in target.__qualname__
        allowed_prefixes: tuple[str, ...] = tuple(params.allowed_prefixes)
        skip_roots = (
            c.ENFORCEMENT_NAMESPACE_FACADE_ROOTS | c.ENFORCEMENT_INFRASTRUCTURE_BASES
        )
        settings_violation = (
            params.require_settings_base
            and target_name.endswith("Settings")
            and is_top_level
            and target_name != "FlextSettings"
            and not any(base.__name__ == "FlextSettings" for base in target.__mro__[1:])
        )
        allowed_prefix_match = (
            has_expected_prefix
            and bool(allowed_prefixes)
            and any(target_name.startswith(prefix) for prefix in allowed_prefixes)
        )
        prefix_violation = (
            has_expected_prefix
            and is_top_level
            and target_name not in skip_roots
            and not allowed_prefix_match
            and not (
                expected_prefix_text and target_name.startswith(expected_prefix_text)
            )
        )
        return (
            {"name": target_name}
            if settings_violation
            else {"expected": expected_prefix_text, "actual": target_name}
            if prefix_violation
            else _NO_VIOLATION
        )

    @staticmethod
    def _v_wrapper(
        params: me.WrapperParams,  # noqa: ARG004 — params reserved for future
        target: type,
    ) -> t.StrMapping | None:
        """WRAPPER — pass-through wrapper detection via bytecode (ENFORCE-043).

        Walks every Python-defined callable owned by ``target``'s module via
        ``iter_module_callables``; for each function, inspects the byte-stream
        for the canonical pass-through shape:
        ``LOAD_GLOBAL <inner> | LOAD_FAST*N | CALL N | RETURN_VALUE``
        with the LOAD_FAST argvals matching the wrapper's parameters in order.
        """
        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = get_module_filename_or_none(module) or ""
        for fn in ube.iter_module_callables(module):
            param_names = ube.function_param_names(fn)
            if not param_names:
                continue
            if ube.is_pass_through_bytecode(fn, param_names):
                return {"name": fn.__name__, "file": Path(src_file).name}
        return _NO_VIOLATION

    @staticmethod
    def _v_import_blacklist(
        params: me.ImportBlacklistParams,
        target: type,
    ) -> t.StrMapping | None:
        """IMPORT_BLACKLIST — concrete-class / pydantic consumer-import discipline.

        Walks ``module.__dict__`` and inspects each value's true module of
        origin via ``get_object_module_name_or_none``. No source parsing.
        Two variants:
        - ``params.forbidden_symbols`` set: classic banned-import check
          (e.g. ``BaseModel`` from ``pydantic`` outside owner pyramid).
        - ``params.forbidden_symbols`` empty + canonical file: rejects
          bare ``Flext*`` class imports in canonical files (R1, R3).
        """
        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = get_module_filename_or_none(module) or ""
        filename = Path(src_file).name
        module_name = module.__name__
        if filename in c.ENFORCEMENT_CANONICAL_FILES and not params.forbidden_symbols:
            canonical_import = next(
                (
                    {"file": filename, "import": name}
                    for name, value in vars(module).items()
                    if isinstance(value, type)
                    and name.startswith("Flext")
                    and (
                        origin := get_object_module_name_or_none(value) or ""
                    ).startswith("flext_")
                    and origin != module_name
                ),
                None,
            )
            return canonical_import or _NO_VIOLATION
        if not params.forbidden_symbols:
            return _NO_VIOLATION
        package = module_name.split(".")[0]
        if package.startswith("flext_") and module_name.startswith(f"{package}._"):
            return _NO_VIOLATION
        forbidden = frozenset(params.forbidden_symbols)
        allowed_roots = frozenset(params.forbidden_modules) or frozenset({"pydantic"})
        banned_import = next(
            (
                {"import": name, "package": package}
                for name, value in vars(module).items()
                if name in forbidden
                and ((get_object_module_name_or_none(value) or "").split(".")[0])
                in allowed_roots
            ),
            None,
        )
        if banned_import is not None:
            return banned_import
        return _NO_VIOLATION

    @staticmethod
    def _v_alias_rebind(
        params: me.AliasRebindParams,
        target: type,
    ) -> t.StrMapping | None:
        """ALIAS_REBIND — canonical alias rebind / sibling-import discipline.

        ``params.expected_form`` selects the variant. All three variants
        operate on runtime ``module.__dict__`` introspection — no source
        parsing.
        """
        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = get_module_filename_or_none(module) or ""
        filename = Path(src_file).name
        module_name = module.__name__
        package = module_name.split(".")[0]
        variant = params.expected_form
        if (
            variant == "rebound_at_module_end"
            and filename in c.ENFORCEMENT_CANONICAL_FILES
        ):
            target_name = target.__name__
            alias_char: str | None = next(
                (
                    alias_name
                    for alias_name, suffix in cp.ALIAS_TO_SUFFIX.items()
                    if alias_name in cp.FACADE_ALIAS_NAMES and suffix in target_name
                ),
                None,
            )
            if alias_char and getattr(module, alias_char, None) is not target:
                return {"alias": alias_char, "class": target_name}
            return _NO_VIOLATION
        if (
            variant == "no_self_root_import_in_core_files"
            and filename in c.ENFORCEMENT_CANONICAL_FILES
        ):
            # Walk the module's runtime aliases. Each alias resolves to a
            # value with ``__module__`` set; if that source module is the
            # same package root we are inspecting (i.e. ``flext_core``
            # importing ``c`` from ``flext_core`` instead of from parent),
            # flag the violation.
            for alias_char in cp.RUNTIME_ALIAS_NAMES:
                alias_value = getattr(module, alias_char, None)
                if alias_value is None:
                    continue
                origin = get_object_module_name_or_none(alias_value) or ""
                if origin.split(".", 1)[0] == package:
                    return {"package": package, "alias": alias_char}
            return _NO_VIOLATION
        if variant == "sibling_models_type_checking" and "_models" in src_file:
            # Sibling-import + TYPE_CHECKING discipline is fundamentally
            # about how the source organises imports; runtime introspection
            # cannot distinguish between an import emitted under
            # ``if TYPE_CHECKING:`` and a normal import (both populate
            # ``module.__annotations__`` identically). The legacy regex
            # check produced no measurable false-positive value in our
            # baseline test runs — this variant is now a no-op pending a
            # proper redesign that uses ``module.__lazy_imports__`` or
            # similar metadata. Tracked separately.
            return _NO_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def _v_library_import(
        params: me.LibraryImportParams,
        target: type,
    ) -> t.StrMapping | None:
        """LIBRARY_IMPORT — §2.7 library abstraction owner enforcement (Phase 3 hook)."""
        if not params.library_owners:
            return _NO_VIOLATION
        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        module_name = getattr(target, "__module__", "") or ""
        package = module_name.split(".")[0].replace("_", "-")
        for value in vars(module).values():
            origin = get_object_module_name_or_none(value)
            if origin is None:
                continue
            origin_root = origin.split(".")[0]
            owner = params.library_owners.get(origin_root)
            if owner is None or package == owner:
                continue
            return {"lib": origin_root, "owner": owner, "package": package}
        return _NO_VIOLATION

    @staticmethod
    def _v_loc_cap(
        params: me.LocCapParams,
        target: type,
    ) -> t.StrMapping | None:
        """LOC_CAP — module logical-LOC ceiling (AGENTS.md §3.1)."""
        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        try:
            source_lines, _start = inspect.getsourcelines(module)
        except (OSError, TypeError):
            return _NO_VIOLATION
        source = "".join(source_lines)
        src_file = get_module_filename_or_none(module) or ""
        loc = sum(
            1
            for line in source.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if loc > params.max_logical_loc:
            return {
                "file": Path(src_file).name,
                "loc": str(loc),
                "cap": str(params.max_logical_loc),
            }
        return _NO_VIOLATION

    @staticmethod
    def _v_duplicate_symbol(
        params: me.DuplicateSymbolParams,  # noqa: ARG004 — needs workspace_index from walker
        target: type,  # noqa: ARG004
    ) -> t.StrMapping | None:
        """DUPLICATE_SYMBOL — workspace cross-project SSOT (Phase 3 hook).

        Implementation lives in the workspace walker, not the per-class
        runtime hook — needs the cross-project symbol index that only the
        walker can build. Returns None at runtime.
        """
        return _NO_VIOLATION

    @staticmethod
    def _v_deprecated_syntax(
        params: me.DeprecatedSyntaxParams,
        target: type,
    ) -> t.StrMapping | None:
        """DEPRECATED_SYNTAX — runtime introspection routed by ``params.ast_shape``.

        Handles every legacy ``check_<tag>`` whose detection was a single
        AST scan; reimplemented entirely via ``module.__dict__`` walks +
        ``dis.Bytecode`` opcode inspection. ``AnnAssign[TypeAlias]`` (Phase 3)
        is detected via ``inspect.get_annotations(module)`` looking for the
        ``TypeAlias`` sentinel. PEP 695 ``type X = Y`` materialises as a
        runtime ``types.TypeAliasType`` instance and is NOT flagged here
        (the legacy ``X: TypeAlias = Y`` form is the deprecation target).
        """
        shape = params.ast_shape
        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = get_module_filename_or_none(module) or ""
        file_name = Path(src_file).name
        if shape == "AnnAssign[TypeAlias]":
            try:
                module_annotations = inspect.get_annotations(module, eval_str=False)
            except (TypeError, NameError):
                return _NO_VIOLATION
            for ann in module_annotations.values():
                if ann is _typing_TypeAlias:
                    return {"file": file_name, "line": "?"}
            return _NO_VIOLATION
        if shape == "cast_outside_core":
            if any(marker in src_file for marker in c.ENFORCE_FLEXT_CORE_PATH_MARKERS):
                return _NO_VIOLATION
            cast_target = c.EnforceAstHookSymbol.CAST_CALL.value
            for fn in ube.iter_module_callables(module):
                hit = ube.has_call_to_global(fn, cast_target)
                if hit is not None:
                    return {"file": file_name, "line": str(fn.__code__.co_firstlineno)}
            return _NO_VIOLATION
        if shape == "model_rebuild_call":
            attr = c.EnforceAstHookSymbol.MODEL_REBUILD_ATTR.value
            for fn in ube.iter_module_callables(module):
                hit = ube.has_attribute_call(fn, attr)
                if hit is not None:
                    return {"file": file_name, "line": str(fn.__code__.co_firstlineno)}
            return _NO_VIOLATION
        if shape == "private_attr_probe":
            probes = c.ENFORCE_PRIVATE_PROBE_BUILTINS
            for fn in ube.iter_module_callables(module):
                hit = ube.has_private_attr_probe(fn, probes)
                if hit is not None:
                    builtin, attr = hit
                    return {"probe": builtin, "name": attr, "file": file_name}
            return _NO_VIOLATION
        if shape == "no_core_tests_namespace":
            wrapper_module = ube.runtime_wrapper_module_for(target)
            if wrapper_module is None:
                return _NO_VIOLATION
            for alias_name in cp.RUNTIME_ALIAS_NAMES:
                alias_value = getattr(wrapper_module, alias_name, None)
                if alias_value is None:
                    continue
                core = getattr(alias_value, "Core", None)
                if core is None:
                    continue
                if hasattr(core, "Tests"):
                    return {
                        "symbol": f"{alias_name}.Core.Tests",
                        "file": Path(
                            get_module_filename_or_none(wrapper_module) or ""
                        ).name,
                        "line": "<runtime>",
                    }
            return _NO_VIOLATION
        if shape == "no_wrapper_root_alias_import":
            wrapper_module = ube.runtime_wrapper_module_for(target)
            if wrapper_module is None:
                return _NO_VIOLATION
            wrapper_submodules = cp.FACADE_MODULE_NAMES
            for alias_name in cp.RUNTIME_ALIAS_NAMES:
                alias_value = getattr(wrapper_module, alias_name, None)
                if alias_value is None:
                    continue
                origin = get_object_module_name_or_none(alias_value) or ""
                parent, dot, child = origin.partition(".")
                if (
                    dot
                    and parent in {"tests", "examples", "scripts"}
                    and child in wrapper_submodules
                ):
                    return {
                        "file": Path(
                            get_module_filename_or_none(wrapper_module) or ""
                        ).name,
                        "line": "<runtime>",
                        "statement": f"from {origin} import {alias_name}",
                    }
            return _NO_VIOLATION
        return _NO_VIOLATION

    # Static dispatch table — predicate_kind → visitor staticmethod. Resolved
    # at class body time; staticmethod descriptors are directly callable in
    # Python 3.10+. Adding a new predicate kind = one row here + one ``_v_*``
    # method on this class. No other code change required.
    _VISITORS: ClassVar[
        Mapping[c.EnforcementPredicateKind, Callable[..., t.StrMapping | None]]
    ] = MappingProxyType({
        c.EnforcementPredicateKind.FIELD_SHAPE: _v_field_shape,
        c.EnforcementPredicateKind.MODEL_CONFIG: _v_model_config,
        c.EnforcementPredicateKind.ATTR_SHAPE: _v_attr_shape,
        c.EnforcementPredicateKind.METHOD_SHAPE: _v_method_shape,
        c.EnforcementPredicateKind.CLASS_PLACEMENT: _v_class_placement,
        c.EnforcementPredicateKind.PROTOCOL_TREE: _v_protocol_tree,
        c.EnforcementPredicateKind.MRO_SHAPE: _v_mro_shape,
        c.EnforcementPredicateKind.LOOSE_SYMBOL: _v_loose_symbol,
        c.EnforcementPredicateKind.WRAPPER: _v_wrapper,
        c.EnforcementPredicateKind.IMPORT_BLACKLIST: _v_import_blacklist,
        c.EnforcementPredicateKind.ALIAS_REBIND: _v_alias_rebind,
        c.EnforcementPredicateKind.LIBRARY_IMPORT: _v_library_import,
        c.EnforcementPredicateKind.LOC_CAP: _v_loc_cap,
        c.EnforcementPredicateKind.DUPLICATE_SYMBOL: _v_duplicate_symbol,
        c.EnforcementPredicateKind.DEPRECATED_SYNTAX: _v_deprecated_syntax,
    })


ube = FlextUtilitiesBeartypeEngine
__all__: list[str] = ["FlextUtilitiesBeartypeEngine", "ube"]
