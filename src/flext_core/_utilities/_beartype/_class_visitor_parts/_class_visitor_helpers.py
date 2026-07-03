"""Private helpers for class placement / MRO / protocol-tree governance."""

from __future__ import annotations

BINARY_ARITY: int = 2
NO_VIOLATION: None = None


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
