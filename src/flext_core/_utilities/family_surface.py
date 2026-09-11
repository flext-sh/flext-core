"""Published FLEXT family-surface derivation from the lazy export contract."""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from .._constants.enforcement import FlextConstantsEnforcement as c
from ..lazy import normalize_lazy_imports

if TYPE_CHECKING:
    from .._typings.base import FlextTypingBase as t

_MIN_PUBLISHED_CONTRACT: Final[int] = 1


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
    def _surface_snapshot() -> (
        tuple[tuple[str, frozenset[str], t.MappingKV[str, t.StrPair | str]], ...]
    ):
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
            snapshot.append(
                (name, frozenset(published), MappingProxyType(dict(normalized))),
            )
        if len(snapshot) < _MIN_PUBLISHED_CONTRACT:
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
            for name, published, _ in (
                FlextUtilitiesFamilySurface._surface_snapshot()
            )
        }
        return MappingProxyType(owners)

    @staticmethod
    def compatibility_alias_renames() -> t.MappingKV[str, str]:
        """Map published long facade class name to its canonical alias.

        Derived by inverting every family root's normalized lazy map for
        single-letter canonical aliases — replacing the frozen per-member
        table and covering every member and published facade automatically.
        Two aliases claiming one long name is a contract violation and fails.
        """
        renames: dict[str, str] = {}
        for _, _, entries in FlextUtilitiesFamilySurface._surface_snapshot():
            for alias, entry in entries.items():
                if len(alias) != 1 or alias not in c.ENFORCEMENT_CANONICAL_ALIASES:
                    continue
                if not isinstance(entry, tuple):
                    continue
                long_name = entry[1]
                if not long_name.startswith("Flext"):
                    continue
                previous = renames.get(long_name)
                if previous is not None and previous != alias:
                    msg = (
                        f"published family surface maps {long_name!r} to both "
                        f"{previous!r} and {alias!r}"
                    )
                    raise ValueError(msg)
                renames[long_name] = alias
        return MappingProxyType(renames)


__all__: list[str] = ["FlextUtilitiesFamilySurface"]
