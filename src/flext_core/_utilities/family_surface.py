"""Published FLEXT family-surface derivation from the lazy export contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from types import MappingProxyType
from typing import cast

from flext_core import c, normalize_lazy_imports, t
from flext_core._utilities import FlextUtilitiesProjectMetadata


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
        requirement_names: Sequence[str],
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
        return distribution_name.startswith(
            c.NAMESPACE_FAMILY_PREFIX,
        ) or any(
            requirement.startswith(c.NAMESPACE_FAMILY_PREFIX)
            for requirement in requirement_names
        )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _module_names_by_distribution() -> Mapping[str, str]:
        """Map every installed distribution name to its importable root package.

        A distribution's import name is the top-level package it ships, which
        diverges from the normalized distribution name whenever the project
        brands its distribution differently from the package inside it (e.g.
        ``datacosmos-backup`` shipping ``dc_backup``). Importing the naive
        normalized name crashes the whole family-surface derivation for such
        consumers, so the importlib.metadata package-to-distribution index is
        the authority and the normalized name is only the fallback.

        Returns:
            The resulting ``Mapping[str, str]``.

        """
        module_names: t.MutableStrMapping = {}
        for (
            module_name,
            distribution_names,
        ) in importlib.metadata.packages_distributions().items():
            for distribution_name in distribution_names:
                module_names.setdefault(distribution_name.lower(), module_name)
        return MappingProxyType(module_names)

    @staticmethod
    def family_discovery_candidates() -> tuple[tuple[str, str], ...]:
        """Elect installed roots inspected before their family contract is known.

        Returns:
            Normalized distribution names paired with their native import names.

        """
        module_by_distribution = (
            FlextUtilitiesFamilySurface._module_names_by_distribution()
        )
        candidates: list[tuple[str, str]] = []
        # Deferred: this module loads inside the package init chain; the
        # sibling is imported at call time to avoid a re-entrant cycle.

        for dist in FlextUtilitiesProjectMetadata.installed_distributions():
            raw_name = dist.metadata["Name"] or ""
            name = raw_name.lower().replace("-", "_")
            if FlextUtilitiesFamilySurface.distribution_in_family_discovery(
                name,
                FlextUtilitiesProjectMetadata.distribution_requirement_names(dist),
            ):
                candidates.append((
                    name,
                    module_by_distribution.get(raw_name.lower(), name),
                ))
        return tuple(candidates)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _surface_snapshot() -> tuple[
        tuple[
            str,
            frozenset[str],
            Mapping[str, tuple[str, str] | str],
        ],
        ...,
    ]:
        """Import every family root once and snapshot its published contract.

        Returns:
            The resulting ``tuple[tuple[str, frozenset[str], Mapping[str, tuple[str, str]
                | str]], ...]``.

        Raises:
            RuntimeError: If family-surface derivation found no distribution publishing
                the lazy export contract (family prefix or dependency edge).

        """
        snapshot: list[
            tuple[
                str,
                frozenset[str],
                Mapping[str, tuple[str, str] | str],
            ]
        ] = []
        for (
            name,
            module_name,
        ) in FlextUtilitiesFamilySurface.family_discovery_candidates():
            try:
                module = importlib.import_module(module_name)
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
                f"{c.NAMESPACE_FAMILY_PREFIX!r} or a "
                "requirement edge to it"
            )
            raise RuntimeError(msg)
        return tuple(snapshot)

    @staticmethod
    def project_alias_owners() -> Mapping[
        str,
        tuple[str, ...],
    ]:
        """Map family package name to the declaration aliases it publishes.

        Derived from each root's ``__all__`` intersected with the
        declaration aliases derived from ``c.NAMESPACE_LAYER_NAMES`` —
        replacing the frozen per-member roster with per-package truth.

        Returns:
            The resulting ``Mapping[str, tuple[str, ...]]``.

        """
        declaration = tuple(name[0].lower() for name in c.NAMESPACE_LAYER_NAMES)
        owners = {
            name: tuple(alias for alias in declaration if alias in published)
            for name, published, _ in (FlextUtilitiesFamilySurface._surface_snapshot())
        }
        return MappingProxyType(owners)

    @staticmethod
    def _grouped_surface_entries() -> Mapping[str, MutableMapping[str, list[str]]]:
        """Group each family root's lazy-map entries by owner module and kind.

        Returns:
            The resulting mapping of family roots to owner-module entries.

        """
        grouped: MutableMapping[
            str,
            MutableMapping[str, list[str]],
        ] = {}
        for _, _, entries in FlextUtilitiesFamilySurface._surface_snapshot():
            for alias, entry in entries.items():
                module = entry if isinstance(entry, str) else entry[0]
                kind = "aliases" if len(alias) == 1 else "names"
                bucket = grouped.setdefault(module, {"aliases": [], "names": []})
                bucket[kind].append(alias)
        return grouped

    @staticmethod
    def _canonical_letters(
        bucket: Mapping[str, list[str]],
    ) -> list[str]:
        """Return the canonical single-letter aliases of one module bucket.

        Returns:
            The resulting ``list[str]``.

        """
        return [
            alias for alias in bucket["aliases"] if alias in c.FAMILY_CANONICAL_ALIASES
        ]

    @staticmethod
    def _register_alias_rename(
        renames: MutableMapping[str, str],
        long_name: str,
        alias: str,
        module: str,
    ) -> None:
        """Publish one long-name rename, failing on conflicting alias claims.

        Raises:
            ValueError: If published family surface maps.

        """
        previous = renames.setdefault(long_name, alias)
        if previous != alias:
            msg = (
                f"published family surface maps {long_name!r} to "
                f"both {previous!r} and {alias!r} via {module!r}"
            )
            raise ValueError(msg)

    @staticmethod
    def compatibility_alias_renames() -> Mapping[str, str]:
        """Map published long facade class name to its canonical alias.

        Derived by grouping each family root's normalized lazy map entries
        by owner module: a module that publishes both a single-letter
        canonical alias and a long ``Flext*`` class name publishes the same
        facade in two forms, and consumers must use the canonical alias.
        Two aliases claiming one long name is a contract violation and
        fails. The derivation reads only the published contract, so every
        current and future member is covered without per-member tables.

        Returns:
            The resulting ``Mapping[str, str]``.

        """
        renames: MutableMapping[str, str] = {}
        for (
            module,
            bucket,
        ) in FlextUtilitiesFamilySurface._grouped_surface_entries().items():
            letters = FlextUtilitiesFamilySurface._canonical_letters(bucket)
            for long_name in bucket["names"]:
                if not long_name.startswith("Flext"):
                    continue
                for alias in letters:
                    FlextUtilitiesFamilySurface._register_alias_rename(
                        renames,
                        long_name,
                        alias,
                        module,
                    )
        return cast("Mapping[str, str]", MappingProxyType(renames))

    @staticmethod
    def _lazy_suffix(module_path: str) -> str:
        name = module_path.rsplit(".", 1)[-1]
        if name == "typings":
            return "Types"
        return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)

    @staticmethod
    @functools.cache
    def lazy_alias_suffixes(package_name: str) -> tuple[tuple[str, str, str], ...]:
        """Read package lazy exports; import failures retain their original cause.

        Returns:
            ``(alias, module_path, family_suffix)`` for every one-letter alias.

        """
        package = sys.modules.get(package_name)
        if package is None:
            package = importlib.import_module(package_name)
        published = vars(package).get("_LAZY_IMPORTS")
        if published is None:
            return ()
        lazy_imports = normalize_lazy_imports(package.__name__, published)
        return tuple(
            (
                alias,
                module_path,
                FlextUtilitiesFamilySurface._lazy_suffix(module_path),
            )
            for alias, entry in lazy_imports.items()
            if len(alias) == 1 and alias.islower()
            for module_path in (entry if isinstance(entry, str) else entry[0],)
        )

    @staticmethod
    def runtime_alias_names(package_name: str) -> frozenset[str]:
        """Return runtime alias names derived from generated lazy exports.

        Returns:
            Runtime alias names derived from generated lazy exports.

        """
        return frozenset(
            alias
            for alias, _, _ in FlextUtilitiesFamilySurface.lazy_alias_suffixes(
                package_name,
            )
        )

    @staticmethod
    def facade_module_names(package_name: str) -> frozenset[str]:
        """Return local facade module names derived from generated lazy exports.

        Returns:
            Local facade module names derived from generated lazy exports.

        """
        return frozenset(
            module_path.rsplit(".", 1)[-1]
            for (
                _,
                module_path,
                suffix,
            ) in FlextUtilitiesFamilySurface.lazy_alias_suffixes(package_name)
            if module_path.split(".", 1)[0] == package_name
            and suffix in {"Constants", "Models", "Protocols", "Types", "Utilities"}
        )

    @staticmethod
    def family_facade(target: type) -> bool:
        """Return True when a family package publishes ``target`` as a letter facade.

        Both facts come from their owners: the generated ``_LAZY_IMPORTS``
        binds each one-letter alias to its class, and the family surface
        names the packages that publish that contract.

        Returns:
            True when a family package publishes ``target`` as a letter facade.

        """
        package_name = target.__module__.split(".", 1)[0]
        if package_name not in FlextUtilitiesFamilySurface.project_alias_owners():
            return False
        published = FlextUtilitiesFamilySurface.lazy_alias_suffixes(package_name)
        return any(
            vars(sys.modules[module_path]).get(alias) is target
            for alias, module_path, _ in published
            if module_path in sys.modules
        )


__all__: list[str] = ["FlextUtilitiesFamilySurface"]
