"""Transformer-based fix adapter for enforcement rules with syntactic rewrites.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m, u
from flext_infra._utilities.rope_imports import FlextInfraUtilitiesRopeImports
from flext_infra.fixers.base import FlextInfraFixerAdapter
from flext_infra.transformers.cast_remover import FlextInfraRefactorCastRemover
from flext_infra.transformers.compatibility_alias import (
    FlextInfraRefactorCompatibilityAlias,
)
from flext_infra.transformers.future_import import FlextInfraRefactorFutureImport
from flext_infra.transformers.hardcoded_version import (
    FlextInfraRefactorHardcodedVersion,
)
from flext_infra.transformers.import_modernizer import (
    FlextInfraRefactorImportModernizer,
)
from flext_infra.transformers.open_encoding import FlextInfraRefactorOpenEncoding
from flext_infra.transformers.pattern import FlextInfraRefactorPatternTransformer
from flext_infra.transformers.project_alias_migrator import (
    FlextInfraRefactorProjectAliasMigrator,
)
from flext_infra.transformers.typing_dict_attr import FlextInfraRefactorTypingDictAttr
from flext_infra.transformers.typing_dict_import import (
    FlextInfraRefactorTypingDictImport,
)
from flext_infra.transformers.typing_unifier import FlextInfraRefactorTypingUnifier

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import p, t
    from flext_infra.transformers.base import FlextInfraRopeTransformer


class FlextInfraTransformerFixerAdapter(FlextInfraFixerAdapter):
    """Apply fixes by running a rope/source transformer per file.

    Targets are canonical transformer short-names declared in the enforcement
    catalog. Each transformer must expose ``apply_to_source(source) -> (str, changes)``.
    """

    kind: ClassVar[str] = "transformer"

    def __init__(self, repository_root: Path) -> None:
        """Bind the repository root used to resolve relative file paths."""
        super().__init__(repository_root)

    # Canonical transformer registry. New deterministic transformers register here.
    _TRANSFORMERS: ClassVar[dict[str, type[FlextInfraRopeTransformer]]] = {
        "cast_remover": FlextInfraRefactorCastRemover,
        "compatibility_alias": FlextInfraRefactorCompatibilityAlias,
        "future_import": FlextInfraRefactorFutureImport,
        "hardcoded_version": FlextInfraRefactorHardcodedVersion,
        "import_modernizer": FlextInfraRefactorImportModernizer,
        "open_encoding": FlextInfraRefactorOpenEncoding,
        "pattern": FlextInfraRefactorPatternTransformer,
        "project_alias_migrator": FlextInfraRefactorProjectAliasMigrator,
        "rewrite_foreign_canonical_alias": FlextInfraRefactorProjectAliasMigrator,
        "typing_dict_attr": FlextInfraRefactorTypingDictAttr,
        "typing_dict_import": FlextInfraRefactorTypingDictImport,
        "typing_unifier": FlextInfraRefactorTypingUnifier,
    }

    @override
    def can_fix(self, fix_action: me.EnforcementFixAction) -> bool:
        """Return whether this adapter handles ``fix_action``."""
        return fix_action.kind == self.kind and fix_action.target in self._TRANSFORMERS

    @override
    def fix_project(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Apply transformer fixes file-by-file for the given violations."""
        if not violations:
            return m.Infra.ProjectFixResult(project=project_dir.name)
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        files_modified: set[str] = set()
        for target, target_violations in self._group_by_target(violations).items():
            transformer_cls = self._TRANSFORMERS.get(target)
            if transformer_cls is None:
                rule_id = self._rule_id(target_violations)
                failed.append(
                    m.Infra.FailedFix(
                        rule_id=rule_id,
                        file_path=str(project_dir),
                        error=f"transformer {target} not registered",
                    )
                )
                continue
            fix_action = target_violations[0][0].fix_action
            file_paths = self._collect_file_paths(project_dir, target_violations)
            for file_path in file_paths:
                if self._is_owned_library_exempt(project_dir, fix_action, file_path):
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=self._rule_id(target_violations),
                            file_path=str(file_path),
                            reason=(
                                f"project {project_dir.name} owns library abstraction"
                            ),
                        )
                    )
                    continue
                result = self._fix_file(
                    file_path=file_path,
                    transformer_cls=transformer_cls,
                    fix_action=fix_action,
                    ctx=ctx,
                    rule_id=self._rule_id(target_violations),
                )
                fixed.extend(result.fixed)
                previewed.extend(result.previewed)
                skipped.extend(result.skipped)
                failed.extend(result.failed)
                files_modified.update(result.files_modified)
        if ctx.apply and files_modified:
            normalize_result = self._normalize_imports(tuple(files_modified))
            if normalize_result.failure:
                failed.append(
                    m.Infra.FailedFix(
                        rule_id="",
                        file_path=str(project_dir),
                        error=normalize_result.error or "import normalization failed",
                    )
                )
        return m.Infra.ProjectFixResult(
            project=project_dir.name,
            fixed=tuple(fixed),
            previewed=tuple(previewed),
            skipped=tuple(skipped),
            failed=tuple(failed),
            files_modified=tuple(files_modified),
        )

    @staticmethod
    def _is_owned_library_exempt(
        project_dir: Path, fix_action: me.EnforcementFixAction | None, file_path: Path
    ) -> bool:
        """Skip import modernization inside the library's owning project.

        Direct imports of pydantic/structlog/oracledb/ldap3 are allowed within
        the project that owns the abstraction facade; consumers must route
        through that facade.
        """
        _ = file_path
        if fix_action is None or fix_action.target != "import_modernizer":
            return False
        imports_to_remove = u.Cli.json_as_sequence(
            fix_action.params.get("imports_to_remove")
        )
        for module in imports_to_remove:
            if not isinstance(module, str):
                continue
            owner = c.ENFORCEMENT_LIBRARY_OWNERS.get(module)
            if owner == project_dir.name:
                return True
        return False

    def _normalize_imports(self, file_paths: t.SequenceOf[str]) -> p.Result[bool]:
        """Run rope+ruff import cleanup on files touched by transformers.

        Keeps canonical runtime-alias imports (c/m/p/t/u) that Ruff may consider
        unused because they are referenced inside string annotations or via
        lazy exports.
        """
        paths = tuple(Path(path) for path in file_paths)
        with u.Infra.open_project(self._repository_root) as rope_project:
            return FlextInfraUtilitiesRopeImports.normalize_imports(
                rope_project, file_paths=paths, preserve_canonical_aliases=True
            )

    def _fix_file(
        self,
        file_path: Path,
        transformer_cls: type[FlextInfraRopeTransformer],
        fix_action: me.EnforcementFixAction | None,
        ctx: m.Infra.FixEnforcementCommand,
        *,
        rule_id: str = "",
    ) -> m.Infra.ProjectFixResult:
        """Run one transformer against one file."""
        if fix_action is None:
            return m.Infra.ProjectFixResult(
                project=file_path.parent.name,
                skipped=(
                    m.Infra.SkippedViolation(
                        rule_id=rule_id,
                        file_path=str(file_path),
                        reason="missing fix_action in catalog",
                    ),
                ),
            )
        read = u.Cli.files_read_text(file_path)
        if read.failure:
            return m.Infra.ProjectFixResult(
                project=file_path.parent.name,
                failed=(
                    m.Infra.FailedFix(
                        rule_id=rule_id,
                        file_path=str(file_path),
                        error=read.error or "unable to read file",
                    ),
                ),
            )
        source = read.value
        transformer = self._build_transformer(
            transformer_cls=transformer_cls, fix_action=fix_action, file_path=file_path
        )
        updated, changes = transformer.apply_to_source(source)
        if not changes:
            return m.Infra.ProjectFixResult(
                project=file_path.parent.name,
                skipped=(
                    m.Infra.SkippedViolation(
                        rule_id=rule_id,
                        file_path=str(file_path),
                        reason="no changes produced",
                    ),
                ),
            )
        if not ctx.apply:
            return m.Infra.ProjectFixResult(
                project=file_path.parent.name,
                previewed=(
                    m.Infra.PreviewedViolation(
                        rule_id=rule_id,
                        file_path=str(file_path),
                        message=f"would apply {len(changes)} change(s)",
                    ),
                ),
            )
        write = u.Cli.files_write_text(file_path, updated)
        if write.failure:
            return m.Infra.ProjectFixResult(
                project=file_path.parent.name,
                failed=(
                    m.Infra.FailedFix(
                        rule_id=rule_id,
                        file_path=str(file_path),
                        error=write.error or "unable to write file",
                    ),
                ),
            )
        return m.Infra.ProjectFixResult(
            project=file_path.parent.name,
            fixed=(
                m.Infra.FixedViolation(
                    rule_id=rule_id,
                    file_path=str(file_path),
                    message=f"applied {len(changes)} change(s)",
                ),
            ),
            files_modified=(str(file_path),),
        )

    @staticmethod
    def _build_transformer(
        transformer_cls: type[FlextInfraRopeTransformer],
        fix_action: me.EnforcementFixAction,
        file_path: Path,
    ) -> FlextInfraRopeTransformer:
        """Instantiate a transformer with params declared in the catalog."""
        params = dict(fix_action.params)
        if transformer_cls is FlextInfraRefactorTypingUnifier:
            targets_value = params.get("targets", [])
            targets: t.StrSequence = (
                tuple(item for item in targets_value if isinstance(item, str))
                if isinstance(targets_value, (list, tuple))
                else ()
            )
            canonical_map: dict[frozenset[str], str] = {}
            if "dict" in targets:
                canonical_map[frozenset({"dict[K, V]"})] = "t.MappingKV[K, V]"
                canonical_map[frozenset({"dict[str, Any]"})] = (
                    "t.MappingKV[str, t.JsonValue]"
                )
            return FlextInfraRefactorTypingUnifier(
                canonical_map=canonical_map, file_path=file_path
            )
        if transformer_cls in {
            FlextInfraRefactorTypingDictImport,
            FlextInfraRefactorTypingDictAttr,
        }:
            return transformer_cls(file_path=file_path)
        if transformer_cls is FlextInfraRefactorProjectAliasMigrator:
            return FlextInfraRefactorProjectAliasMigrator(file_path=file_path)
        if transformer_cls is FlextInfraRefactorImportModernizer:
            imports_to_remove = tuple(
                name
                for name in u.Cli.json_as_sequence(params.get("imports_to_remove"))
                if isinstance(name, str)
            )
            symbols_to_replace = {
                k: str(v)
                for k, v in u.Cli.json_as_mapping(
                    params.get("symbols_to_replace")
                ).items()
                # flext-i6nq.10: Mapping keys are already typed as strings.
                if isinstance(v, (str, int, float))
            }
            runtime_aliases = {
                name
                for name in u.Cli.json_as_sequence(params.get("runtime_aliases"))
                if isinstance(name, str)
            }
            blocked_aliases = {
                name
                for name in u.Cli.json_as_sequence(params.get("blocked_aliases"))
                if isinstance(name, str)
            }
            return FlextInfraRefactorImportModernizer(
                imports_to_remove=imports_to_remove,
                symbols_to_replace=symbols_to_replace,
                runtime_aliases=runtime_aliases,
                blocked_aliases=blocked_aliases,
            )
        if transformer_cls is FlextInfraRefactorPatternTransformer:
            required_alias = params.get("required_alias", "")
            alias_module = params.get("alias_module", "")
            return FlextInfraRefactorPatternTransformer(
                patterns=u.Cli.json_as_mapping_list(params.get("patterns")),
                required_alias=required_alias
                if isinstance(required_alias, str)
                else "",
                alias_module=alias_module if isinstance(alias_module, str) else "",
                file_path=file_path,
            )
        # Remaining enforcement transformers require no runtime params.
        return transformer_cls()


__all__: list[str] = ["FlextInfraTransformerFixerAdapter"]
