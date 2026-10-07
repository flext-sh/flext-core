"""MRO_SHAPE alias/peer-first analysis sidecar.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

from flext_core._constants import FlextConstantsEnforcement
from flext_core._models import FlextModelsEnforcement
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities import (
    FlextUtilitiesBeartypeHelpers,
    FlextUtilitiesProjectMetadata,
)
from flext_core._utilities._beartype._class_visitor_parts.class_visitor_part_01 import (
    BINARY_ARITY,
    NO_VIOLATION,
)

type AliasRows = tuple[tuple[str, str, str], ...]
type AliasContext = tuple[
    str,
    str,
    AliasRows,
    tuple[str, ...],
    tuple[str, ...],
    bool,
]
type FacadeScope = tuple[bool, bool, bool]
type BaseAnalysis = tuple[int, str, str, bool, bool]


def _alias_context(target: type) -> AliasContext | None:
    """Resolve the module/package alias context, or ``None`` when exempt.

    Private family classes (e.g. FlextApiConstantsApi in _constants/api.py)
    are composed by their parent facade through MRO (R1, R3).  The
    alias/peer-first checks (ENFORCE-047, ENFORCE-049) only constrain
    public facade classes, so private family modules are exempt.
    The family set is driven by the ENFORCEMENT_PRIVATE_FAMILY_PACKAGES
    constant (see FlextConstantsEnforcementTargets) — evaluation iterates
    that set AND any private (underscore-prefixed) sub-package path
    component, so adding a new private sub-package is automatic.

    Returns:
        The resulting ``AliasContext | None``.

    """
    _, separator, _ = target.__qualname__.partition(".")
    is_module_level = not separator
    module_name = getattr(target, "__module__", "") or ""
    package_name = module_name.split(".", 1)[0]
    project_prefix: str = target.__name__
    if target.__module__:
        project_prefix = FlextUtilitiesProjectMetadata.derive_class_stem(package_name)
    tier_facade_prefixes = (project_prefix, f"Tests{project_prefix}")
    # A facade lives in its package's alias modules, and the import that created
    # a real class already loaded its package; a class whose package is not
    # loaded is synthetic and declares no facade.
    alias_rows: AliasRows = (
        FlextUtilitiesBeartypeHelpers.lazy_alias_suffixes(package_name)
        if package_name in sys.modules
        else ()
    )
    suffixes = tuple(suffix for _, _, suffix in alias_rows)
    valid_suffixes = suffixes + tuple(f"{suffix}Base" for suffix in suffixes)
    module_parts = module_name.split(".")
    if len(module_parts) > 1 and (
        module_parts[1] in FlextConstantsEnforcement.ENFORCEMENT_PRIVATE_FAMILY_PACKAGES
        or any(part.startswith("_") for part in module_parts[1:])
    ):
        return None
    return (
        module_name,
        package_name,
        alias_rows,
        valid_suffixes,
        tier_facade_prefixes,
        is_module_level,
    )


def _facade_scope(
    target: type,
    params: FlextModelsEnforcement.MroShapeParams,
    ctx: AliasContext,
) -> FacadeScope:
    """Resolve the (require_alias_first, is_facade, is_core_root) scope.

    Returns:
        The resulting ``FacadeScope``.

    """
    module_name, _, alias_rows, _, tier_facade_prefixes, is_module_level = ctx
    is_facade = all((
        is_module_level,
        target.__name__.startswith(tier_facade_prefixes),
        any(module_name == module_path for _, module_path, _ in alias_rows),
    ))
    excluded_roots = (
        "flext_core.tests",
        "flext_core.examples",
        "flext_core.scripts",
    )
    is_core_root = module_name.startswith("flext_core.") and (
        not module_name.startswith(excluded_roots)
    )
    return (params.require_alias_first, is_facade, is_core_root)


def _peer_first_allowed(
    *,
    base_count: int,
    names: tuple[str, str],
    scope: tuple[bool, bool],
    suffixes: tuple[tuple[str, ...], tuple[str, ...]],
    shared_peer_alias_base: set[type],
) -> bool:
    """Return True when a facade may place a peer base first.

    Returns:
        True when a facade may place a peer base first.

    """
    is_facade, is_core_root = scope
    first_name, unparametrized_name = names
    valid_suffixes, tier_facade_prefixes = suffixes
    if not is_facade or is_core_root or base_count < BINARY_ARITY:
        return False
    if not first_name.startswith(tier_facade_prefixes):
        return False
    if unparametrized_name.endswith(valid_suffixes):
        return False
    return bool(shared_peer_alias_base)


def _requires_alias_first(
    *,
    scope: FacadeScope,
    names: tuple[str, str],
    suffixes: tuple[tuple[str, ...], tuple[str, ...]],
    is_alias_or_alias_base_first: bool,
    allows_peer_first: bool,
) -> bool:
    """Return True when a facade base must be an alias/alias-base first.

    Returns:
        True when a facade base must be an alias/alias-base first.

    """
    require_alias_first, is_facade, is_core_root = scope
    _, unparametrized_name = names
    valid_suffixes, _ = suffixes
    if not require_alias_first or not is_facade or is_core_root:
        return False
    if is_alias_or_alias_base_first:
        return False
    if unparametrized_name.endswith(valid_suffixes):
        return False
    return not allows_peer_first


def _alias_base_sets(
    target: type,
    valid_suffixes: tuple[str, ...],
) -> tuple[list[set[type]], set[type]]:
    """Collect per-base alias ancestors and their shared intersection.

    Returns:
        The resulting ``(alias_base_sets, shared_peer_alias_base)`` pair.

    """
    alias_base_sets = [
        {
            ancestor
            for ancestor in base.__mro__[1:]
            if getattr(ancestor, "__name__", "").split("[")[0].endswith(valid_suffixes)
        }
        for base in target.__bases__
    ]
    peer_alias_bases = [base_set for base_set in alias_base_sets if base_set]
    shared_peer_alias_base = (
        set.intersection(*peer_alias_bases) if peer_alias_bases else set()
    )
    return (alias_base_sets, shared_peer_alias_base)


def _is_service_alias_base_first(
    first_base: type,
    first_name: str,
    package_name: str,
    tier_facade_prefixes: tuple[str, ...],
) -> bool:
    """Return True when the first base is the package's FlextService alias.

    Returns:
        The resulting ``bool``.

    """
    first_base_package = first_base.__module__.split(".", 1)[0]
    return all((
        first_base_package == package_name,
        first_name.startswith(tier_facade_prefixes),
        any(
            getattr(ancestor, "__name__", "").split("[")[0] == "FlextService"
            for ancestor in first_base.__mro__[1:]
        ),
    ))


def _base_analysis(target: type, ctx: AliasContext, scope: FacadeScope) -> BaseAnalysis:
    """Analyze the target bases for alias/peer ordering.

    Returns:
        The resulting ``BaseAnalysis`` of ``(base_count, first_name,
        unparametrized_name, is_alias_or_alias_base_first,
        allows_peer_first)``.

    """
    _, package_name, _, valid_suffixes, tier_facade_prefixes, _ = ctx
    base_count = len(target.__bases__)
    first_name = getattr(target.__bases__[0], "__name__", "")
    # Strip generic parameters so ``FlextService[T]`` → ``FlextService``
    unparametrized_name = first_name.split("[")[0]
    alias_base_sets, shared_peer_alias_base = _alias_base_sets(target, valid_suffixes)
    is_service_alias_base_first = _is_service_alias_base_first(
        target.__bases__[0],
        first_name,
        package_name,
        tier_facade_prefixes,
    )
    # `FlextService[T]` specializations are the canonical core service root
    # for facade packages and should not be treated as a missing alias.
    is_alias_or_alias_base_first = (
        unparametrized_name == "FlextService"
        or unparametrized_name.endswith(valid_suffixes)
        or is_service_alias_base_first
    )
    allows_single_peer_base = all((
        base_count == 1,
        first_name.startswith(tier_facade_prefixes),
        not unparametrized_name.endswith(valid_suffixes),
        bool(alias_base_sets[0]) if alias_base_sets else False,
    ))
    _, is_facade, is_core_root = scope
    allows_peer_first = (is_facade and allows_single_peer_base) or (
        _peer_first_allowed(
            base_count=base_count,
            names=(first_name, unparametrized_name),
            scope=(is_facade, is_core_root),
            suffixes=(valid_suffixes, tier_facade_prefixes),
            shared_peer_alias_base=shared_peer_alias_base,
        )
    )
    return (
        base_count,
        first_name,
        unparametrized_name,
        is_alias_or_alias_base_first,
        allows_peer_first,
    )


def _violation_payload(
    scope: FacadeScope,
    ctx: AliasContext,
    analysis: BaseAnalysis,
) -> FlextTypingBase.StrMapping | None:
    """Select the alias-first violation payload for the analyzed bases.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    _, _, _, valid_suffixes, tier_facade_prefixes, _ = ctx
    base_count, first_name, unparam_name, is_alias_first, allows_peer_first = analysis
    requires_alias_first = _requires_alias_first(
        scope=scope,
        names=(first_name, unparam_name),
        suffixes=(valid_suffixes, tier_facade_prefixes),
        is_alias_or_alias_base_first=is_alias_first,
        allows_peer_first=allows_peer_first,
    )
    min_multi_parent = 2
    return next(
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


def alias_first_violation(
    target: type,
    params: FlextModelsEnforcement.MroShapeParams,
) -> FlextTypingBase.StrMapping | None:
    """Compute the alias/peer-first violation for ``v_mro_shape``.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    ctx = _alias_context(target)
    if ctx is None:
        return NO_VIOLATION
    scope = _facade_scope(target, params, ctx)
    analysis = _base_analysis(target, ctx, scope)
    return _violation_payload(scope, ctx, analysis)


__all__: list[str] = ["alias_first_violation"]
