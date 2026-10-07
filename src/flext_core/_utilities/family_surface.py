"""Published FLEXT family-surface derivation from the lazy export contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._utilities.project_metadata import FlextUtilitiesProjectMetadata
from flext_core.lazy import normalize_lazy_imports

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesFamilySurface:
    """Derive the published FLEXT family surface at runtime.

    Membership grammar (declared, never enumerated): a distribution whose
    normalized name starts with ``c.NAMESPACE_FAMILY_PREFIX`` — or that
    declares a requirement on one — is a family member when its root package
    publishes the lazy export contract (``__all__`` plus ``_LAZY_IMPORTS``).
    The prefix and the requirement edge only narrow discovery; the published
    contract is the structural proof, so every current and future member is
    covered with zero per-member registration. A consumer facade that
    publishes the contract (a Pattern-A project composing the family through
    facade classes) is discovered by its dependency edge, so its facade
    classes count as family facades and subclassing a family facade there is
    the sanctioned shape. The first import failure escapes — broken installs
    are defects, never skips.
    """

    @staticmethod
    def distribution_in_family_discovery(
        distribution_name: str,
        requirement_names: t.StrSequence,
    ) -> bool:
        """Return True when a distribution is a family discovery candidate.

        A candidate either carries the family prefix in its own normalized
        name or declares a requirement on one; discovery is membership's
        narrow gate, and the published lazy export contract remains the
        structural proof a candidate must still pass.

        Returns:
            True when the distribution name or one requirement carries the
            family prefix.
        """
        return distribution_name.startswith(c.NAMESPACE_FAMILY_PREFIX) or any(
            requirement.startswith(c.NAMESPACE_FAMILY_PREFIX)
            for requirement in requirement_names
        )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _module_names_by_distribution() -> t.MappingKV[str, str]:
        """Map every installed distribution name to its importable root package.

        A distribution's import name is the top-level package it ships, which
        diverges from the normalized distribution name whenever the project
        brands its distribution differently from the package inside it (e.g.
        ``datacosmos-backup`` shipping ``dc_backup``). Importing the naive
        normalized name crashes the whole family-surface derivation for such
        consumers, so the importlib.metadata package-to-distribution index is
        the authority and the normalized name is only the fallback.

        Returns:
            The resulting ``t.MappingKV[str, str]``.

        """
        module_names: dict[str, str] = {}
        for (
            module_name,
            distribution_names,
        ) in importlib.metadata.packages_distributions().items():
            for distribution_name in distribution_names:
                module_names.setdefault(distribution_name.lower(), module_name)
        return MappingProxyType(module_names)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _surface_snapshot() -> tuple[
        tuple[str, frozenset[str], t.MappingKV[str, t.StrPair | str]],
        ...,
    ]:
        """Import every family root once and snapshot its published contract.

        Returns:
            The resulting ``tuple[tuple[str, frozenset[str], t.MappingKV[str, t.StrPair
                | str]], ...]``.

        Raises:
            RuntimeError: If family-surface derivation found no distribution publishing
                the lazy export contract (family prefix or dependency edge).

        """
        snapshot: list[
            tuple[str, frozenset[str], t.MappingKV[str, t.StrPair | str]]
        ] = []
        module_by_distribution = (
            FlextUtilitiesFamilySurface._module_names_by_distribution()
        )
        for dist in FlextUtilitiesProjectMetadata.installed_distributions():
            raw_name = dist.metadata["Name"] or ""
            name = raw_name.lower().replace("-", "_")
            if not FlextUtilitiesFamilySurface.distribution_in_family_discovery(
                name,
                FlextUtilitiesProjectMetadata.distribution_requirement_names(dist),
            ):
                continue
            try:
                module = importlib.import_module(
                    module_by_distribution.get(raw_name.lower(), name),
                )
            except ImportError:
                # The requirement edge only nominates the member; the published
                # contract is the structural proof. A member whose root package
                # cannot be imported under any derivable name (an editable
                # install whose RECORD names no top-level package, for one)
                # publishes nothing this snapshot can read — it is not a
                # surface failure, and one such member must not crash the
                # derivation every other consumer relies on.
                continue
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
                f"{c.NAMESPACE_FAMILY_PREFIX!r} or a requirement edge to it"
            )
            raise RuntimeError(msg)
        return tuple(snapshot)

    @staticmethod
    def project_alias_owners() -> t.MappingKV[str, t.VariadicTuple[str]]:
        """Map family package name to the declaration aliases it publishes.

        Derived from each root's ``__all__`` intersected with the
        declaration aliases derived from ``c.NAMESPACE_LAYER_NAMES`` —
        replacing the frozen per-member roster with per-package truth.

        Returns:
            The resulting ``t.MappingKV[str, t.VariadicTuple[str]]``.

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

        Returns:
            The resulting ``t.MappingKV[str, str]``.

        Raises:
            ValueError: If published family surface maps.

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
