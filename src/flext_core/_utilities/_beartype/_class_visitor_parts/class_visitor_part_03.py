"""Class placement, MRO, and protocol tree governance."""

from __future__ import annotations

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t

from .class_visitor_part_01 import (
    NO_VIOLATION,
)
from .class_visitor_part_02 import (
    FlextUtilitiesBeartypeClassVisitor as FlextUtilitiesBeartypeClassVisitorPart02,
)


class FlextUtilitiesBeartypeClassVisitor(FlextUtilitiesBeartypeClassVisitorPart02):
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
                return NO_VIOLATION
        target_name = target.__name__
        is_top_level = "." not in target.__qualname__
        allowed_prefixes: t.StrSequence = tuple(params.allowed_prefixes)
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
            else NO_VIOLATION
        )


__all__: list[str] = ["FlextUtilitiesBeartypeClassVisitor"]
