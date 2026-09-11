"""Census object classification, violation building, and impact map — extracted concern."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, m

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraRefactorCensusObjectsMixin:
    """Object classification, violation/key construction, and impact-map results.

    Composed into FlextInfraRefactorCensus via inheritance; self-contained
    (static/cls helpers over report Object/Violation models, no census state).
    """

    @staticmethod
    def _raw_violation(
        *,
        project: str,
        object_name: str,
        object_kind: str,
        kind: str,
        file_path: Path,
        line: int,
        description: str,
        fixable: bool = False,
        fix_action: str = "",
    ) -> m.Infra.Census.Violation:
        """Raw violation."""
        return m.Infra.Census.Violation(
            project=project,
            object_name=object_name,
            object_kind=object_kind,
            kind=kind,
            file_path=str(file_path),
            line=line,
            fixable=fixable,
            fix_action=fix_action,
            description=description,
        )

    @staticmethod
    def _selected_families(family_names: t.StrSequence | None) -> frozenset[str]:
        """Return the selected families."""
        if not family_names:
            return frozenset()
        resolved = {
            c.Infra.FAMILY_SUFFIXES.get(name, name).lower() for name in family_names
        }
        return frozenset(resolved)

    @staticmethod
    def _violation(
        item: m.Infra.Census.Object,
        *,
        kind: str,
        description: str,
        fixable: bool = False,
        fix_action: str = "",
    ) -> m.Infra.Census.Violation:
        """Violation."""
        return FlextInfraRefactorCensusObjectsMixin._raw_violation(
            project=item.project,
            object_name=item.name,
            object_kind=item.kind,
            kind=kind,
            file_path=Path(item.file_path),
            line=item.line,
            description=description,
            fixable=fixable,
            fix_action=fix_action,
        )

    @staticmethod
    def _is_unused(item: m.Infra.Census.Object) -> bool:
        """Is unused."""
        return (
            not item.is_facade_member
            and item.references_count == 0
            and not item.name.startswith("_")
        )

    @classmethod
    def _removal_candidate(
        cls, item: m.Infra.Census.Object, *, include_unused: bool
    ) -> m.Infra.Census.RemovalCandidate | None:
        """Build a removal candidate for an object."""
        if include_unused and cls._is_unused(item):
            reason, suggested_action = "unused", "delete_object_definition"
        else:
            return None
        return m.Infra.Census.RemovalCandidate(
            project=item.project,
            object_name=item.name,
            object_kind=item.kind,
            file_path=item.file_path,
            line=item.line,
            scope_path=item.scope_path,
            reason=reason,
            suggested_action=suggested_action,
            runtime_reference_sites=item.runtime_reference_sites,
            script_reference_sites=item.script_reference_sites,
        )

    @staticmethod
    def _object_key(item: m.Infra.Census.Object) -> str:
        """Object key."""
        return f"{item.file_path}:{item.line}:{item.scope_path}:{item.kind}"

    @staticmethod
    def _fix_key(file_path: Path, object_name: str, action: str = "") -> str:
        """Fix key."""
        suffix = f"::{action}" if action else ""
        return f"{file_path.resolve()}::{object_name}{suffix}"

    @classmethod
    def _impact_map_results(
        cls, report: m.Infra.Census.WorkspaceReport
    ) -> tuple[m.Infra.Result, ...]:
        """Impact map results."""
        changes_by_file: dict[Path, list[str]] = defaultdict(list)
        for candidate in report.removal_candidates:
            source_path = Path(candidate.file_path)
            cls._append_impact_change(
                changes_by_file,
                source_path,
                f"{candidate.suggested_action}: {candidate.object_name} ({candidate.reason})",
            )
            for site in cls._reference_sites(candidate):
                cls._append_impact_change(
                    changes_by_file,
                    Path(site.file_path),
                    f"remove reference to {candidate.object_name} at line {site.line} ({site.surface})",
                )
        return tuple(
            m.Infra.Result(
                file_path=file_path,
                success=True,
                modified=True,
                changes=tuple(changes_by_file[file_path]),
            )
            for file_path in sorted(changes_by_file, key=Path.as_posix)
        )

    @staticmethod
    def _append_impact_change(
        changes_by_file: dict[Path, list[str]], file_path: Path, change: str
    ) -> None:
        """Append impact change."""
        normalized_path = file_path.resolve()
        if change not in changes_by_file[normalized_path]:
            changes_by_file[normalized_path].append(change)

    @staticmethod
    def _reference_sites(
        candidate: m.Infra.Census.RemovalCandidate,
    ) -> tuple[m.Infra.Census.ReferenceSite, ...]:
        """Return all reference sites for a removal candidate."""
        return (*candidate.runtime_reference_sites, *candidate.script_reference_sites)

    @staticmethod
    def _is_flext_owned(value: p.ModuleOwned) -> bool:
        """Return True iff `value`'s defining module is in the flext package tree.

        Used to filter the parent inventory so that builtin attributes
        inherited by str/int/dict/list constants do not pollute collision
        candidates with names like `count`, `index`, `replace`, etc.
        """
        module_name = getattr(value, "__module__", "")
        if not isinstance(module_name, str):
            return False
        return module_name.startswith(c.Infra.PKG_PREFIX_UNDERSCORE)


__all__: list[str] = ["FlextInfraRefactorCensusObjectsMixin"]
