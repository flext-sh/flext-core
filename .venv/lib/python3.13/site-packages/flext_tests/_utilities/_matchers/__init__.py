# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Utilities. Matchers package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._assertions import FlextTestsMatchersAssertionsMixin
    from ._containment import FlextTestsMatchersContainmentMixin
    from ._immutability import FlextTestsMatchersImmutabilityMixin
    from ._result import FlextTestsMatchersResultMixin
    from ._scope import FlextTestsMatchersScopeMixin
    from ._that import FlextTestsMatchersThatMixin
    from ._typeguards import FlextTestsMatchersTypeGuardsMixin
__all__: tuple[str, ...] = (
    "FlextTestsMatchersAssertionsMixin",
    "FlextTestsMatchersContainmentMixin",
    "FlextTestsMatchersImmutabilityMixin",
    "FlextTestsMatchersResultMixin",
    "FlextTestsMatchersScopeMixin",
    "FlextTestsMatchersThatMixin",
    "FlextTestsMatchersTypeGuardsMixin",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._assertions": ("FlextTestsMatchersAssertionsMixin",),
            "._containment": ("FlextTestsMatchersContainmentMixin",),
            "._immutability": ("FlextTestsMatchersImmutabilityMixin",),
            "._result": ("FlextTestsMatchersResultMixin",),
            "._scope": ("FlextTestsMatchersScopeMixin",),
            "._that": ("FlextTestsMatchersThatMixin",),
            "._typeguards": ("FlextTestsMatchersTypeGuardsMixin",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
