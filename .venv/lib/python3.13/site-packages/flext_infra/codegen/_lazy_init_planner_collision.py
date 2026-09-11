"""Export-collision resolution for the lazy-init planner."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraCodegenLazyInitPlannerCollisionMixin:
    if TYPE_CHECKING:
        rope_workspace: p.Infra.RopeWorkspaceDsl
        lazy_init: m.Infra.LazyInitConfig
        _collision_count: int

        def _module_file(self, module_path: str) -> Path | None: ...

    def _target_score(self, name: str, target: t.StrPair) -> int:
        """Score a candidate export target to break collision ties."""
        module_path, attr = target
        score = 0
        module_file = self._module_file(module_path)
        if module_file is None:
            return score
        convention = self.rope_workspace.convention(module_file)
        policy = convention.module_policy
        if policy.expected_alias == name:
            score += 100
        elif policy.expected_family and name.endswith(policy.expected_family):
            # A governed facade legitimately owns its declared family class
            # (e.g. ``FlextTestsValidator`` in ``validator.py``); it must win
            # over an FLEXT ``_part_`` module of the same name.
            score += 25
        elif policy.expected_alias:
            # Governed root facades should primarily own their canonical alias.
            score -= 40
        elif policy.expected_family:
            # Penalize cross-family leakage from governed facade files.
            score -= 20
        if policy.export_symbols:
            score += 20
        if policy.enforce_contract:
            score += 10
        declared_exports = self.rope_workspace.exports(
            module_file,
            export_options=m.Infra.ExportOptions(
                allow_assignments=True, allow_functions=True, require_explicit_all=True
            ),
        )
        if name in declared_exports:
            score += 15
        if attr == name:
            score += 3
        part_number = module_file.stem.rpartition("_part_")[2]
        if part_number.isdecimal():
            # flext-pulj (codex): the final public facade owns the external
            # class identity; numbered implementation parts only rank among
            # themselves when no facade candidate exists.
            score -= 50
            score += int(part_number)
        score -= module_path.count(".")
        final_score: int = score
        return final_score

    def _pick_preferred_target(
        self, name: str, existing: t.StrPair, target: t.StrPair
    ) -> t.StrPair:
        """Return the higher-scored of two competing export targets."""
        existing_score = self._target_score(name, existing)
        target_score = self._target_score(name, target)
        if target_score > existing_score:
            return target
        if target_score < existing_score:
            return existing
        return min(existing, target)

    def _add(self, index: t.MutableLazyAliasMap, name: str, target: t.StrPair) -> None:
        """Insert a name/target pair, resolving collisions via policy scoring."""
        existing = index.get(name)
        if existing is None or existing == target:
            index[name] = target
            return
        # flext-j47u (codex): MutableLazyAliasMap values are always StrPair.
        winner = self._pick_preferred_target(name, existing, target)
        if self._is_intentional_reexport(existing, target):
            index[name] = winner
            return
        self._collision_count += 1
        u.Cli.warning(
            f"export collision for {name!r}: {existing} vs {target}; "
            f"resolved by canonical policy scorer to {winner}"
        )
        index[name] = winner

    def _is_intentional_reexport(self, a: t.StrPair, b: t.StrPair) -> bool:
        """Return whether one module is a root-namespace stub re-exporting from the other."""
        if self._is_flext_part_reexport(a, b):
            return True
        # flext-pulj (codex): root typing sidecars are removed; real source
        # owners now participate in the normal collision policy.
        if self._is_private_facade_reexport(a, b):
            return True
        if self._is_declared_public_reexport(a, b):
            return True
        if self._is_test_collection_collision(a, b):
            return True
        for pub_mod, priv_mod in ((a[0], b[0]), (b[0], a[0])):
            pub_file = f"{pub_mod.rsplit('.', maxsplit=1)[-1]}.py"
            if not u.Infra.matches_root_namespace_file(pub_file):
                continue
            if "." in priv_mod and priv_mod.split(".")[-2].startswith("_"):
                return True
        return False

    def _is_declared_public_reexport(self, a: t.StrPair, b: t.StrPair) -> bool:
        if a[1] != b[1]:
            return False
        a_owner = self._declared_export_owner(a)
        b_owner = self._declared_export_owner(b)
        return (
            a_owner == b or b_owner == a or (a_owner is not None and a_owner == b_owner)
        )

    def _declared_export_owner(self, target: t.StrPair) -> t.StrPair | None:
        module_path, attr = target
        module_file = self._module_file(module_path)
        if module_file is None or module_file.name == "__init__.py":
            return None
        tree = ast.parse(module_file.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            package_parts = module_path.split(".")[:-1]
            if node.level:
                retained_parts = package_parts[: len(package_parts) - node.level + 1]
                imported_parts = node.module.split(".") if node.module else []
                imported_module = ".".join((*retained_parts, *imported_parts))
            elif node.module:
                imported_module = node.module
            else:
                continue
            for alias in node.names:
                exported_name = alias.asname or alias.name
                if exported_name == attr:
                    owner_module = (
                        f"{imported_module}.{alias.name}"
                        if node.level and node.module is None
                        else imported_module
                    )
                    return (owner_module, alias.name)
        return None

    @staticmethod
    def _module_parts(module_path: str) -> tuple[str, ...]:
        """Return normalized dotted module path parts."""
        return tuple(part for part in module_path.split(".") if part)

    @staticmethod
    def _part_family_index(parts: t.StrSequence) -> int:
        """Return the index of the private ``*_parts`` package segment."""
        for index, part in enumerate(parts[:-1]):
            if part.startswith("_") and part.endswith("_parts"):
                return index
        return -1

    @staticmethod
    def _is_part_leaf(module_stem: str) -> bool:
        """Return whether a module stem is an FLEXT implementation part."""
        return "_part_" in module_stem

    @classmethod
    def _is_flext_part_reexport(cls, a: t.StrPair, b: t.StrPair) -> bool:
        """Return whether targets are the same logical FLEXT owner split into parts."""
        if a[1] != b[1]:
            return False
        a_parts = cls._module_parts(a[0])
        b_parts = cls._module_parts(b[0])
        a_index = cls._part_family_index(a_parts)
        b_index = cls._part_family_index(b_parts)
        if (
            cls._is_part_leaf(a_parts[-1])
            and cls._is_part_leaf(b_parts[-1])
            and a_parts[:-1] == b_parts[:-1]
        ):
            return True
        if a_index < 0 and b_index < 0:
            return False
        if a_index >= 0 and b_index >= 0:
            a_family: tuple[str, ...] = tuple(a_parts[: a_index + 1])
            b_family: tuple[str, ...] = tuple(b_parts[: b_index + 1])
            return a_family == b_family
        part_parts, part_index, facade_parts = (
            (a_parts, a_index, b_parts) if a_index >= 0 else (b_parts, b_index, a_parts)
        )
        owner_package = part_parts[:part_index]
        if facade_parts[:-1] == owner_package:
            return True
        if not owner_package or not owner_package[-1].startswith("_"):
            return False
        expected_facade_parts: tuple[str, ...] = (
            *tuple(owner_package[:-1]),
            owner_package[-1].removeprefix("_"),
        )
        facade_tuple: tuple[str, ...] = tuple(facade_parts)
        return facade_tuple == expected_facade_parts

    @classmethod
    def _is_private_facade_reexport(cls, a: t.StrPair, b: t.StrPair) -> bool:
        """Return whether a public facade re-exports a private implementation module."""
        if a[1] != b[1]:
            return False
        for pub_mod, priv_mod in ((a[0], b[0]), (b[0], a[0])):
            pub_parts = cls._module_parts(pub_mod)
            priv_parts = cls._module_parts(priv_mod)
            if not pub_parts or not priv_parts or pub_parts[0] != priv_parts[0]:
                continue
            pub_private_segments = cls._private_segments(pub_parts)
            priv_private_segments = cls._private_segments(priv_parts)
            if not (priv_private_segments - pub_private_segments):
                continue
            if pub_parts[:-1] == priv_parts[:-1]:
                return True
            if pub_parts[-1] == priv_parts[-1].removeprefix("_"):
                return True
        return False

    @classmethod
    def _private_segments(cls, parts: t.StrSequence) -> frozenset[tuple[int, str]]:
        """Return private implementation segments with their path positions."""
        return frozenset(
            (index, part)
            for index, part in enumerate(parts[1:], start=1)
            if part.startswith("_") and not part.startswith("__")
        )

    @classmethod
    def _is_test_collection_collision(cls, a: t.StrPair, b: t.StrPair) -> bool:
        """Return whether duplicate generated test collection names are benign."""
        if a[1] != b[1] or not a[1].startswith("Tests"):
            return False
        a_parts = cls._module_parts(a[0])
        b_parts = cls._module_parts(b[0])
        return bool(a_parts and b_parts and a_parts[0] == b_parts[0] == "tests")

    @staticmethod
    def _publish(name: str, *, allow_main: bool) -> bool:
        """Return whether a name should be published in the lazy __init__."""
        return (
            not name.startswith("_")
            and name not in c.Infra.INFRA_ONLY_EXPORTS
            and name not in {c.Infra.DUNDER_INIT, "pytestmark"}
            and (name != "main" or allow_main)
        )
