"""Class placement, MRO, and protocol tree governance."""

from __future__ import annotations

from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers as ubh

from .class_visitor_part_01 import (
    BINARY_ARITY,
    NO_VIOLATION,
    FlextUtilitiesBeartypeClassVisitor as FlextUtilitiesBeartypeClassVisitorPart01,
)


class FlextUtilitiesBeartypeClassVisitor(FlextUtilitiesBeartypeClassVisitorPart01):
    @staticmethod
    def v_mro_shape(
        params: me.MroShapeParams,
        target: type,
    ) -> t.StrMapping | None:
        """MRO_SHAPE — facade base ordering and inner-namespace redundancy."""
        if not target.__bases__:
            return NO_VIOLATION

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
            base_count >= BINARY_ARITY,
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
            NO_VIOLATION,
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
            {"class": target.__qualname__} if has_redundant_inner else NO_VIOLATION
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
        normalized_values = (
            value.__func__ if isinstance(value, (classmethod, staticmethod)) else value
            for value in vars(target).values()
        )
        self_ref_violation = (
            {"class": target.__name__, "first_base": "u"}
            if all((
                violation is None,
                src_file.endswith("utilities.py"),
                not getattr(target, "__flext_pattern_b__", False),
                base_count >= min_multi_parent,
                first_name == "u",
            ))
            and any(
                (code := getattr(value, "__code__", None)) is not None
                and "u" in code.co_names
                for value in normalized_values
                if callable(value)
            )
            else NO_VIOLATION
        )
        return violation or self_ref_violation


__all__: list[str] = ["FlextUtilitiesBeartypeClassVisitor"]
