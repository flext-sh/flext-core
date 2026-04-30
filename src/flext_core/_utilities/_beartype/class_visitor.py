"""Class placement, MRO, and protocol tree governance."""

from __future__ import annotations

from enum import EnumType

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.helpers import (
    FlextUtilitiesBeartypeHelpers as ubh,
)

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}
_BINARY_ARITY: int = 2


class FlextUtilitiesBeartypeClassVisitor:
    """CLASS_PLACEMENT + PROTOCOL_TREE + MRO_SHAPE + LOOSE_SYMBOL visitors."""

    @staticmethod
    def v_class_placement(
        params: me.ClassPlacementParams,
        *args: type | str,
    ) -> t.StrMapping | None:
        """CLASS_PLACEMENT — class-name / inner-class layer placement."""
        violation = _NO_VIOLATION
        match args:
            case (value, layer) if (
                isinstance(value, type)
                and isinstance(layer, str)
                and layer in c.ENFORCEMENT_LAYER_ALLOWS
            ):
                allowed = c.ENFORCEMENT_LAYER_ALLOWS.get(layer, frozenset())
                forbidden_base_matches = (
                    ("StrEnum", isinstance(value, EnumType)),
                    ("Protocol", ubh.has_runtime_protocol_marker(value)),
                )
                if any(
                    base_name in params.forbidden_bases
                    and is_match
                    and base_name not in allowed
                    for base_name, is_match in forbidden_base_matches
                ):
                    violation = _BARE_VIOLATION
            case (target, expected) if isinstance(target, type) and isinstance(
                expected, str
            ):
                if params.check_nested:
                    parts = target.__qualname__.split(".")
                    has_wrong_nested_prefix = all((
                        len(parts) >= c.ENFORCEMENT_NESTED_MRO_MIN_DEPTH,
                        not parts[0].startswith(expected),
                    ))
                    violation = (
                        {"expected": expected}
                        if has_wrong_nested_prefix
                        else _NO_VIOLATION
                    )
                else:
                    violation = (
                        {"expected": expected, "actual": target.__name__}
                        if not target.__name__.startswith(expected)
                        else _NO_VIOLATION
                    )
            case _:
                pass
        return violation

    @staticmethod
    def v_protocol_tree(
        params: me.ProtocolTreeParams,
        value: type,
    ) -> t.StrMapping | None:
        """PROTOCOL_TREE — inner-class kind + runtime_checkable governance."""
        if params.require_inner_kind_protocol_or_namespace:
            if (
                ubh.has_runtime_protocol_marker(value)
                or ubh.has_nested_namespace(value)
                or ubh.has_abstract_contract(value)
            ):
                pass
            else:
                return _BARE_VIOLATION
        if (
            params.require_runtime_checkable
            and ubh.has_runtime_protocol_marker(value)
            and not getattr(value, "_is_runtime_protocol", False)
        ):
            return _BARE_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def v_mro_shape(
        params: me.MroShapeParams,
        target: type,
    ) -> t.StrMapping | None:
        """MRO_SHAPE — facade base ordering and inner-namespace redundancy."""
        if not target.__bases__:
            return _NO_VIOLATION

        outer_name, separator, _ = target.__qualname__.partition(".")
        is_module_level = not separator
        project_prefix, _ = target.__name__, ""
        if target.__module__:
            package_name = target.__module__.split(".", 1)[0]
            project_prefix = mpm.derive_class_stem(package_name)
        tier_facade_prefixes = (project_prefix, f"Tests{project_prefix}")
        is_facade = is_module_level and target.__name__.startswith(tier_facade_prefixes)
        module_name = getattr(target, "__module__", "") or ""
        is_core_root = module_name.startswith(
            "flext_core."
        ) and not module_name.startswith((
            "flext_core.tests",
            "flext_core.examples",
            "flext_core.scripts",
        ))

        base_count = len(target.__bases__)
        first_base = target.__bases__[0]
        first_name = getattr(first_base, "__name__", "")
        # Strip generic parameters so ``FlextService[T]`` → ``FlextService``
        unparametrized_name = first_name.split("[")[0]
        package_name = module_name.split(".", 1)[0]
        suffixes = tuple(
            suffix for _, _, suffix in ubh.lazy_alias_suffixes(package_name)
        )
        valid_suffixes = suffixes + tuple(f"{suffix}Base" for suffix in suffixes)
        alias_base_sets = [
            {
                ancestor
                for ancestor in base.__mro__[1:]
                if getattr(ancestor, "__name__", "")
                .split("[")[0]
                .endswith(valid_suffixes)
            }
            for base in target.__bases__
        ]
        peer_alias_bases = [base_set for base_set in alias_base_sets if base_set]
        shared_peer_alias_base = (
            set.intersection(*peer_alias_bases) if peer_alias_bases else set()
        )
        first_base_package = first_base.__module__.split(".", 1)[0]
        is_service_alias_base_first = all((
            first_base_package == package_name,
            first_name.startswith(tier_facade_prefixes),
            any(
                getattr(ancestor, "__name__", "").split("[")[0] == "FlextService"
                for ancestor in first_base.__mro__[1:]
            ),
        ))
        # `FlextService[T]` specializations are the canonical core service root
        # for facade packages and should not be treated as a missing alias.
        is_alias_or_alias_base_first = (
            unparametrized_name == "FlextService"
            or unparametrized_name.endswith(valid_suffixes)
            or is_service_alias_base_first
        )
        allows_single_peer_base = all((
            is_facade,
            not is_core_root,
            base_count == 1,
            first_name.startswith(tier_facade_prefixes),
            not unparametrized_name.endswith(valid_suffixes),
            bool(alias_base_sets[0]) if alias_base_sets else False,
        ))
        allows_peer_first = allows_single_peer_base or all((
            is_facade,
            not is_core_root,
            base_count >= _BINARY_ARITY,
            first_name.startswith(tier_facade_prefixes),
            not unparametrized_name.endswith(valid_suffixes),
            bool(shared_peer_alias_base),
        ))
        requires_alias_first = (
            params.require_alias_first
            and is_facade
            and not is_core_root
            and not is_alias_or_alias_base_first
            and not unparametrized_name.endswith(valid_suffixes)
            and not allows_peer_first
        )
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
                        requires_alias_first
                        and first_name.startswith(tier_facade_prefixes),
                        {
                            "base": first_name,
                            "expected": "alias, alias-base, or FlextPeerXxx",
                        },
                    ),
                )
                if enabled
            ),
            _NO_VIOLATION,
        )
        outer_name, separator, _ = target.__qualname__.partition(".")
        has_only_dunder_attrs = all(
            key.startswith("__") and key.endswith("__") for key in vars(target)
        )
        has_redundant_inner = all((
            alias_violation is None,
            params.forbid_redundant_inner,
            bool(separator),
            getattr(first_base, "__qualname__", "") == outer_name,
            has_only_dunder_attrs,
        ))
        redundant_inner_violation = (
            {"class": target.__qualname__} if has_redundant_inner else _NO_VIOLATION
        )
        violation = alias_violation or redundant_inner_violation
        maybe_module = (
            ubh.runtime_module_for(target)
            if violation is None and params.require_explicit_class_when_self_ref
            else None
        )
        if maybe_module is None:
            return violation

        module = maybe_module
        src_file = str(getattr(module, "__file__", "") or "")
        package = module.__name__.split(".")[0]
        normalized_values = (
            value.__func__ if isinstance(value, (classmethod, staticmethod)) else value
            for value in vars(target).values()
        )
        self_ref_violation = (
            {"class": target.__name__, "first_base": "u"}
            if all((
                violation is None,
                src_file.endswith("utilities.py"),
                package not in c.ENFORCEMENT_PATTERN_B_UTILITIES_WHITELIST,
                base_count >= min_multi_parent,
                first_name == "u",
            ))
            and any(
                (code := getattr(value, "__code__", None)) is not None
                and "u" in code.co_names
                for value in normalized_values
                if callable(value)
            )
            else _NO_VIOLATION
        )
        return violation or self_ref_violation

    @staticmethod
    def v_loose_symbol(
        params: me.LooseSymbolParams,
        *args: type | str,
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
        inherits_flext_settings = any(
            base.__name__ == "FlextSettings" for base in target.__mro__[1:]
        )
        is_settings_target = all((
            params.require_settings_base,
            target_name.endswith("Settings"),
            is_top_level,
            target_name != "FlextSettings",
        ))
        settings_violation = is_settings_target and not inherits_flext_settings
        allowed_prefix_match = (
            has_expected_prefix
            and bool(allowed_prefixes)
            and any(target_name.startswith(prefix) for prefix in allowed_prefixes)
        )
        has_expected_named_prefix = bool(
            expected_prefix_text
        ) and target_name.startswith(expected_prefix_text)
        is_prefixed_target = all((
            has_expected_prefix,
            is_top_level,
            target_name not in skip_roots,
        ))
        prefix_violation = (
            is_prefixed_target
            and not allowed_prefix_match
            and not has_expected_named_prefix
        )
        return (
            {"name": target_name}
            if settings_violation
            else {"expected": expected_prefix_text, "actual": target_name}
            if prefix_violation
            else _NO_VIOLATION
        )
