"""MRO_SHAPE alias/peer-first analysis sidecar."""

from __future__ import annotations

from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype._class_visitor_parts.class_visitor_part_01 import (
    BINARY_ARITY,
    NO_VIOLATION,
)
from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers as ubh


def _peer_first_allowed(
    *,
    is_facade: bool,
    is_core_root: bool,
    base_count: int,
    first_name: str,
    unparametrized_name: str,
    valid_suffixes: tuple[str, ...],
    tier_facade_prefixes: tuple[str, ...],
    shared_peer_alias_base: set[type],
) -> bool:
    """Return True when a facade may place a peer base first."""
    if not is_facade or is_core_root or base_count < BINARY_ARITY:
        return False
    if not first_name.startswith(tier_facade_prefixes):
        return False
    if unparametrized_name.endswith(valid_suffixes):
        return False
    return bool(shared_peer_alias_base)


def _requires_alias_first(
    *,
    require_alias_first: bool,
    is_facade: bool,
    is_core_root: bool,
    is_alias_or_alias_base_first: bool,
    unparametrized_name: str,
    valid_suffixes: tuple[str, ...],
    allows_peer_first: bool,
) -> bool:
    """Return True when a facade base must be an alias/alias-base first."""
    if not require_alias_first or not is_facade or is_core_root:
        return False
    if is_alias_or_alias_base_first:
        return False
    if unparametrized_name.endswith(valid_suffixes):
        return False
    return not allows_peer_first


def alias_first_violation(
    target: type,
    params: me.MroShapeParams,
) -> t.StrMapping | None:
    """Compute the alias/peer-first violation for ``v_mro_shape``."""
    _, separator, _ = target.__qualname__.partition(".")
    is_module_level = not separator
    project_prefix, _ = target.__name__, ""
    if target.__module__:
        package_name = target.__module__.split(".", 1)[0]
        project_prefix = mpm.derive_class_stem(package_name)
    tier_facade_prefixes = (project_prefix, f"Tests{project_prefix}")
    is_facade = is_module_level and target.__name__.startswith(tier_facade_prefixes)
    module_name = getattr(target, "__module__", "") or ""
    is_core_root = module_name.startswith(
        "flext_core.",
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
    suffixes = tuple(suffix for _, _, suffix in ubh.lazy_alias_suffixes(package_name))
    valid_suffixes = suffixes + tuple(f"{suffix}Base" for suffix in suffixes)
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
    allows_peer_first = allows_single_peer_base or _peer_first_allowed(
        is_facade=is_facade,
        is_core_root=is_core_root,
        base_count=base_count,
        first_name=first_name,
        unparametrized_name=unparametrized_name,
        valid_suffixes=valid_suffixes,
        tier_facade_prefixes=tier_facade_prefixes,
        shared_peer_alias_base=shared_peer_alias_base,
    )
    requires_alias_first = _requires_alias_first(
        require_alias_first=params.require_alias_first,
        is_facade=is_facade,
        is_core_root=is_core_root,
        is_alias_or_alias_base_first=is_alias_or_alias_base_first,
        unparametrized_name=unparametrized_name,
        valid_suffixes=valid_suffixes,
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
