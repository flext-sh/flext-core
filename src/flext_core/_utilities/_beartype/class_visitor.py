"""Class placement, MRO, and protocol tree governance."""

from __future__ import annotations

from enum import EnumType

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cp
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}
_BINARY_ARITY: int = 2


class FlextUtilitiesBeartypeClassVisitor:
    """CLASS_PLACEMENT + PROTOCOL_TREE + MRO_SHAPE + LOOSE_SYMBOL visitors."""

    @staticmethod
    def _v_class_placement(
        params: me.ClassPlacementParams,
        *args: object,
    ) -> t.StrMapping | None:
        """CLASS_PLACEMENT — class-name / inner-class layer placement."""
        from flext_core._utilities.beartype_engine import ube

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
        from flext_core._utilities.beartype_engine import ube

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
        from beartype._util.func.utilfunccodeobj import (
            get_func_code_object_or_none,
        )
        from beartype._util.func.utilfunctest import is_func_python
        from flext_core._utilities.beartype_engine import ube

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
        if module is None:
            return violation
        from beartype._util.module.utilmodget import get_module_filename_or_none

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
        """LOOSE_SYMBOL — top-level class/function naming + settings inheritance."""
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
