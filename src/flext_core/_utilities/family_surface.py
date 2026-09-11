"""Published FLEXT family-surface derivation from the lazy export contract."""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from .._constants.enforcement import FlextConstantsEnforcement as c
from ..lazy import normalize_lazy_imports

if TYPE_CHECKING:
    from .._typings.base import FlextTypingBase as t


class FlextUtilitiesFamilySurface:
    """Derive the published FLEXT family surface at runtime.

    Membership grammar (declared, never enumerated): a distribution whose
    normalized name starts with ``c.NAMESPACE_FAMILY_PREFIX`` is a family
    member when its root package publishes the lazy export contract
    (``__all__`` plus ``_LAZY_IMPORTS``). The prefix only narrows discovery;
    the published contract is the structural proof, so every current and
    future member is covered with zero per-member registration. The first
    import failure escapes — broken installs are defects, never skips.
    """

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _surface_snapshot() -> tuple[
        tuple[str, frozenset[str], t.MappingKV[str, t.StrPair | str]], ...
    ]:
        """Import every family root once and snapshot its published contract."""
        snapshot: list[
            tuple[str, frozenset[str], t.MappingKV[str, t.StrPair | str]]
        ] = []
        for dist in importlib.metadata.distributions():
            raw_name = dist.metadata["Name"] or ""
            name = raw_name.lower().replace("-", "_")
            if not name.startswith(c.NAMESPACE_FAMILY_PREFIX):
                continue
            module = importlib.import_module(name)
            published = getattr(module, "__all__", None)
            raw_map = vars(module).get("_LAZY_IMPORTS")
            if published is None or raw_map is None:
                continue
            normalized = normalize_lazy_imports(module.__name__, raw_map)
            snapshot.append((
                name,
                frozenset(published),
                MappingProxyType(dict(normalized)),
            ))
        if len(snapshot) < c.FAMILY_SURFACE_MIN_PUBLISHED:
            msg = (
                "family-surface derivation found no distribution publishing "
                "the lazy export contract under prefix "
                f"{c.NAMESPACE_FAMILY_PREFIX!r}"
            )
            raise RuntimeError(msg)
        return tuple(snapshot)

    @staticmethod
    def project_alias_owners() -> t.MappingKV[str, tuple[str, ...]]:
        """Map family package name to the declaration aliases it publishes.

        Derived from each root's ``__all__`` intersected with the
        declaration aliases derived from ``c.NAMESPACE_LAYER_NAMES`` —
        replacing the frozen per-member roster with per-package truth.
        """
        declaration = tuple(name[0].lower() for name in c.NAMESPACE_LAYER_NAMES)
        owners = {
            name: tuple(alias for alias in declaration if alias in published)
            for name, published, _ in (FlextUtilitiesFamilySurface._surface_snapshot())
        }
        return MappingProxyType(owners)

    @staticmethod
    def compatibility_alias_renames() -> t.MappingKV[str, str]:
        """Map published long facade class name to its canonical alias.

        Derived by grouping each family root's normalized lazy map entries
        by owner module: a module that publishes both a single-letter
        canonical alias and a long ``Flext*`` class name publishes the same
        facade in two forms, and consumers must use the canonical alias.
        Two aliases claiming one long name is a contract violation and
        fails. The derivation reads only the published contract, so every
        current and future member is covered without per-member tables.
        """
        grouped = {}
        for _, _, entries in FlextUtilitiesFamilySurface._surface_snapshot():
            for alias, entry in entries.items():
                module = entry if isinstance(entry, str) else entry[0]
                kind = "aliases" if len(alias) == 1 else "names"
                if module not in grouped:
                    grouped[module] = cast(
                        "t.MutableMappingKV[str, list[str]]",
                        {"aliases": [], "names": []},
                    )
                bucket = grouped[module]
                bucket[kind].append(alias)
        renames: t.MutableMappingKV[str, str] = {}
        for module, bucket in grouped.items():
            letters = [
                alias
                for alias in bucket["aliases"]
                if alias in c.ENFORCEMENT_CANONICAL_ALIASES
            ]
            for long_name in bucket["names"]:
                if not long_name.startswith("Flext"):
                    continue
                for alias in letters:
                    previous = renames.setdefault(long_name, alias)
                    if previous != alias:
                        msg = (
                            f"published family surface maps {long_name!r} to "
                            f"both {previous!r} and {alias!r} via {module!r}"
                        )
                        raise ValueError(msg)
        return cast("t.StrMapping", MappingProxyType(renames))


__all__: list[str] = ["FlextUtilitiesFamilySurface"]
